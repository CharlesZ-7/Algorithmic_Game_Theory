from adx.agents import NDaysNCampaignsAgent
from adx.tier1_ndays_ncampaign_agent import Tier1NDaysNCampaignsAgent
from adx.adx_game_simulator import AdXGameSimulator
from adx.structures import Bid, Campaign, BidBundle, MarketSegment
from typing import Set, Dict, List
import itertools
import pulp

class User:
    """
    Represents a market_segments
    """
    def __init__(self, market_segment, nums, reserve_price = 0):
        self.market_segment = market_segment
        self.nums = nums
        self.reserve_price = reserve_price

class Solution:
    
    def greedy_allocation(self, campaigns_on_going:  Set[Campaign], impressions: Set[User]) -> Dict[str, Dict[str, int]]:
        """
        Given a market, returns a greedy allocation.
        input: on-going campaigns, impressions
        Output: allocation
        """
        remaining_supply = {user.market_segment: user.nums for user in impressions}
        allocation = {campaign.uid: {user.market_segment: 0 for user in impressions} for campaign in campaigns_on_going}
        total_allocation = {campaign.uid: 0 for campaign in campaigns_on_going}
        
        # Loop through Campaigns
        for campaign in campaigns_on_going:
            # Check if there are enough goods to completely allocate the campaign.
            total_users = sum([remaining_supply[user.market_segment] for user in impressions if user.market_segment.issubset(campaign.target_segment)])
            if  total_users >= campaign.reach:
                # Loop through Goods.
                for user_market_segment in remaining_supply.keys():
                    if user_market_segment.issubset(campaign.target_segment):
                        allocation[campaign.uid][user_market_segment] = min(campaign.reach - total_allocation[campaign.uid], remaining_supply[user_market_segment])
                        total_allocation[campaign.uid] = total_allocation[campaign.uid] + allocation[campaign.uid][user_market_segment]
                        remaining_supply[user_market_segment] = remaining_supply[user_market_segment] - allocation[campaign.uid][user_market_segment]
                # If the goods are too expensive, give them back
                if sum([allocation[campaign.uid][user.market_segment] * user.reserve_price for user in impressions]) > campaign.budget:
                    total_allocation[campaign.uid] = 0
                    for user in impressions:
                        allocation[campaign.uid][user.market_segment] = 0
                        remaining_supply[user.market_segment] += allocation[campaign.uid][user.market_segment]
        return allocation

    def pricing_policy(self, campaigns_on_going: Set[Campaign], impressions: Set[User], allocation: Dict[str, Dict[str, int]]) -> Dict[str, float]:
        """
        Given an allocation, computes prices.
        Input: an allocated market
        Output: a dictionary with prices, one per user
        """
        allocation_users = {}
        allocation_campaigns = {}
        
        for campaign in campaigns_on_going:
            for user in impressions:
                if allocation.get(campaign.uid, {}).get(user.market_segment, 0) != 0:
                    allocation_users[user] = allocation_users.get(user, 0) + allocation[campaign.uid][user.market_segment]
                    allocation_campaigns[campaign] = allocation_campaigns.get(campaign, 0) + allocation[campaign.uid][user.market_segment]
                    
        prices_variables = pulp.LpVariable.dicts('prices', [user.market_segment for user in allocation_users.keys()], 0.0)
        indifference_slack_variables = pulp.LpVariable.dicts('slack', [(campaign, user1, user2) for campaign, user1, user2 in itertools.product(allocation_campaigns.keys(), allocation_users.keys(), allocation_users.keys())], 0.0)
    
        model = pulp.LpProblem("Pricing", pulp.LpMaximize)
        model += pulp.lpSum([prices_variables[user.market_segment] * allocation_users[user] for user in allocation_users.keys()] + [-var for var in indifference_slack_variables.values()])
    
        for campaign in allocation_campaigns.keys():
            if allocation_campaigns[campaign] > 0:
                model += pulp.lpSum([prices_variables[user.market_segment] * allocation[campaign.uid][user.market_segment] for user in allocation_users.keys() if user.market_segment in allocation[campaign.uid]]) <= campaign.budget
        
        for user in allocation_users:
            model += prices_variables[user.market_segment] >= user.reserve_price
            for campaign in allocation_campaigns:
                for other_user in allocation_users:
                    if user != other_user and other_user.market_segment.issubset((campaign.target_segment)):
                        model += prices_variables[user.market_segment] <= prices_variables[other_user.market_segment] + indifference_slack_variables[(campaign, user, other_user)]
                        
        model.solve()
        
        return {user.market_segment: prices_variables[user.market_segment].varValue for user in allocation_users.keys()}

    def walrasian_equilibrium(self, campaigns_on_going: Set[Campaign], impressions: Set[User]) -> List[float]:
        # The we strategy is to bid only on those goods for which the bidder was allocated.
        # The bid is (bid, limit) = (p_g, p_g x_cg) in case p_g >0; otherwise the bid is (bid, limit) = (0.0, c.budget) in case p_g=0.
        allocation = self.greedy_allocation(campaigns_on_going, impressions)
        prices = self.pricing_policy(campaigns_on_going, impressions, allocation)
        return [(campaign.uid, user.market_segment, prices[user.market_segment], allocation[campaign.uid][user.market_segment] * prices[user.market_segment] if prices[user.market_segment] > 0 else campaign.budget)
                for campaign in campaigns_on_going for user in impressions if allocation[campaign.uid][user.market_segment] > 0]

