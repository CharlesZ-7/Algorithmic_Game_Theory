from adx.agents import NDaysNCampaignsAgent
from adx.tier1_ndays_ncampaign_agent import Tier1NDaysNCampaignsAgent
from adx.adx_game_simulator import AdXGameSimulator
from adx.structures import Bid, Campaign, BidBundle, MarketSegment
from typing import Set, Dict
import itertools
import pulp

class User:
    """
    Represents a demography
    """
    def __init__(self, market_segment, nums, reserved_price = 0):
        self.market_segment = market_segment
        self.nums = nums
        self.reserved_price = reserved_price

market_segments = [
    MarketSegment(("Male", "Young", "LowIncome")),
    MarketSegment(("Male", "Young", "HighIncome")),
    MarketSegment(("Male", "Old", "LowIncome")),
    MarketSegment(("Male", "Old", "HighIncome")),
    MarketSegment(("Female", "Young", "LowIncome")),
    MarketSegment(("Female", "Young", "HighIncome")),
    MarketSegment(("Female", "Old", "LowIncome")),
    MarketSegment(("Female", "Old", "HighIncome"))
]

class MyNDaysNCampaignsAgent(NDaysNCampaignsAgent):

    def __init__(self):
        # TODO: fill this in (if necessary)
        super().__init__()
        self.name = "My Agent"  # TODO: enter a name.

    def on_new_game(self) -> None:
        # TODO: fill this in (if necessary)
        pass

    def greedy_algorithm(self, campaigns_on_going:  Set[Campaign], impressions: Set[User]) -> List[List[int]]:
        """
        Given a market, returns a greedy allocation.
        input: on-going campaigns, impressions
        Output: allocation
        """
        remaining_supply = {user: user.num for user in impressions}
        allocation = {campaign: {user: 0 for user in impressions} for campaign in campaigns_on_going}
        total_allocation = {campaign: 0 for campaign in campaigns_on_going}
        
        # Loop through Campaigns
        for campaign in campaigns_on_going:
            # Check if there are enough goods to completely allocate the campaign.
            if sum([remaining_supply[demography] for user in impressions if user.market_segment in campaign.target_segment]) >= campaign.reach
                # Loop through Goods.
                for user in supply.keys():
                    if user.market_segment in campaign.target_segment:
                        allocation[campaign][user] = min(campaign.reach - total_allocation[campaign], remaining_supply[demography])
                        total_allocation[campaign] = total_allocation[campaign] + allocation[campaign][user]
                        remaining_supply[user] = remaining_supply[user] - allocation[campaign][user]
                # If the goods are too expensive, give them back
                if sum([allocation[campaign][user] * demography.reserve_price for user in impressions]) > campaign.budget:
                    total_allocation[campaign] = 0
                    for user in impressions:
                        allocation[campaign][user] = 0
                        remaining_supply[user] += allocation[campaign][user]
        return allocation
        
    def pricing_policy(self, campaigns_on_going:  Set[Campaign], impressions: Set[User], allocation: List[List[int]]) -> Dict[User, float]:
        """
        Given an allocation, computes prices.
        imput: an allocated market
        output: a dictionary with prices, one per user
        """
        # Construct the Lp. We need a variable per good.
        set_of_allocated_campaigns = set()
        set_of_assigned_users = set()
        for campaign in campaigns_on_going:
            for user in impressions:
                if allocation[campaign][user]:
                    set_of_allocated_campaigns.add(campaign)
                    set_of_assigned_users.add(user)
                    
        list_of_allocated_campaigns = list(set_of_allocated_campaigns)
        list_of_assigned_users = list(set_of_assigned_users)
                    
        prices_variables = pulp.LpVariable.dicts('prices', list_of_assigned_users, 0.0)
        indifference_slack_variables = pulp.LpVariable.dicts('slack', [(campaign, user1, user2) for campaign, user1, user2 in itertools.product(list_of_allocated_campaigns, list_of_assigned_users, list_of_assigned_users)], 0.0)
        model = pulp.LpProblem("Pricing", pulp.LpMaximize)
        # Maximum revenue objective, minimizing slack.
        for user in list_of_assigned_users:
            total_user_allocation = 0
            for campaign in list_of_allocated_campaigns:
                total_user_allocation += allocation[campaign][user]
            model += prices_variables[user] * total_user_allocation
        for _, indifference_slack_var in indifference_slack_variables.items():
            model -= indifference_slack_var
            
        # IR constraints.
        for campaign in list_of_allocated_campaigns:
            total_campaign_allocation = 0
            for user in list_of_assigned_users:
                total_campaign_allocation += allocation[campaign][user]
            if total_campaign_allocation > 0:
                model += sum([prices_variables[user] * allocation[campaign][user] for user in list_of_assigned_users]) <= campaign.budget
        # Indifference condition, a.k.a. compact condition, relaxed with slacks
        for user in list_of_assigned_users:
            # Reserve price constraints
            model += prices_variables[user] >= user.reserve_price
            for campaign in list_of_allocated_campaign:
                if allocation[campaign][user] > 0:
                    for other_user in allocation.market.goods:
                        if other_user != user and other_user.market_segment in campaign.target_segment and allocation[campaign][other_user] < other_user.supply:
                            model += prices_variables[user] <= prices_variables[other_user] + indifference_slack_variables[(campaign, user, other_user)]
        model.solve()
        # for _, indifference_slack_var in indifference_slack_variables.items():
        #    if indifference_slack_var.value() > 0 or True:
        #        print(indifference_slack_var.value())
        return {user: prices_variables[user].value() for user in list_of_assigned_users}

    def walrasian_equilibrium(self, campaigns_on_going:  Set[Campaign], impressions: Set[User]) -> List[float]:
        # The we strategy is to bid only on those goods for which the bidder was allocated.
        # The bid is (bid, limit) = (p_g, p_g x_cg) in case p_g >0; otherwise the bid is (bid, limit) = (0.0, c.budget) in case p_g=0.
        allocation = greedy_allocation(campaigns_on_going, impressions)
        prices = pricing_policy(allocation)
        return [(campaign, user, prices[user], allocation[campaign][user] * prices[user] if prices[user] > 0 else campaign.budget)
                for campaign in campaigns_on_going for user in impressions if allocation[campaign][user] > 0]

    def get_ad_bids(self) -> Set[BidBundle]:
        # TODO: fill this in
        
        bundles = set()

        # notes:

        # if there are two campaigns with two auctions...
            # which does it prioritize...
            # take into consideration the profit...
            # take into account the effects on the quality score...

            # prioritize the campaign with lower effective reach...

        # during the last days of the campaign, we don't care about the quality score...

        # ... 



        # --- basic syntax to create and send bids ---

        # all campaigns, campaigns from auction and random campaigns
        campaigns = self.get_active_campaigns().union(self.my_campaigns)

        # iterate over campaigns
        for campaign in campaigns:

            # create bids
            bid_entries = set()

            # find all subsets
            subsets = set()
            for segment in MarketSegment.all_segments():
                if campaign.target_segment.issuperset(segment):
                    subsets.add(segment)

            # iterate over subsets
            n = len(subsets)
            for segment in subsets:
                
                # auction items (market segment) # this is a market segment...
                auction_item = segment
                
                # bid per item
                bid_per_item = campaign.budget / (n * campaign.reach)

                # create bid object
                bid = Bid(
                    bidder=self,
                    auction_item=auction_item,
                    bid_per_item=bid_per_item,
                    bid_limit=campaign.budget / n
                )
                bid_entries.add(bid)

            # set spending limit 
            limit = campaign.budget

            # bundle bid and return
            bundle = BidBundle(
                campaign_id=campaign.uid,
                limit=limit,
                bid_entries=bid_entries
            )
            bundles.add(bundle)

        return bundles

    def get_campaign_bids(self, campaigns_for_auction:  Set[Campaign]) -> Dict[Campaign, float]:
        # TODO: fill this in 
        
            
        bids = {}

        # notes:

        # if agent already owns a campaign, should it go for a monopoly or diversify?
            # try to win campaigns for similar market sections to block competition
            # don't get too many campaigns otherwise quality scores will go down

        # assign campaigns some kind of score for how much we want to bid on it...

        # try to prioritize shorter campaigns... retain high effective reach...
            # just be sure that it is still possible to fufill...

        # if we know what other campaigns are going on, go for the less competitive ones...

        # how low to bid on an auction as this determines the budget for that auction...
            # some kind of ai here?? we need better ai's 
            # some kind of history...
            # lenght of contracts... fill these as fast as we can...

        # ... 



        # --- basic syntax to create and send bids ---

        # iterate over campaigns
        for campaign in campaigns_for_auction:
            bids[campaign] = 1000.

        return bids

if __name__ == "__main__":
    # Here's an opportunity to test offline against some TA agents. Just run this file to do so.
    test_agents = [MyNDaysNCampaignsAgent()] + [Tier1NDaysNCampaignsAgent(name=f"Agent {i + 1}") for i in range(9)]

    # Don't change this. Adapt initialization to your environment
    simulator = AdXGameSimulator()
    simulator.run_simulation(agents=test_agents, num_simulations=1) # originally 500 