
import gymnasium as gym
import numpy as np
import random
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict
from math import isfinite, atan

import sys, os
sys.path.append("..")
# sys.path.append(os.path.abspath(os.path.join(os.path.pardir, 'adx')))
# sys.path.append(sys.path[0] + "/../adx")

from adx.structures import Campaign, Bid, BidBundle, MarketSegment
from adx.pmfs import PMF
from adx.tier1_ndays_ncampaign_agent import Tier1NDaysNCampaignsAgent
from adx.states import CampaignBidderState
from adx.agents import NDaysNCampaignsAgent
from waterfall_algorithm import waterfall
from training_agents import *

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

CONFIG = {
        'num_agents': 10,
        'num_days': 10,
        'quality_score_alpha': 0.5,
        'campaigns_per_day': 5,
        'campaign_reach_dist': [0.3,0.5,0.7],
        'campaign_length_dist': [1, 2, 3],
        'market_segment_dist': [
            MarketSegment(("Male", "Young")),
            MarketSegment(("Male", "Old")),
            MarketSegment(("Male", "LowIncome")),
            MarketSegment(("Male", "HighIncome")),
            MarketSegment(("Female", "Young")),
            MarketSegment(("Female", "Old")),
            MarketSegment(("Female", "LowIncome")),
            MarketSegment(("Female", "HighIncome")),
            MarketSegment(("Young", "LowIncome")),
            MarketSegment(("Young", "HighIncome")),
            MarketSegment(("Old", "LowIncome")),
            MarketSegment(("Old", "HighIncome")),
            MarketSegment(("Male", "Young", "LowIncome")),
            MarketSegment(("Male", "Young", "HighIncome")),
            MarketSegment(("Male", "Old", "LowIncome")),
            MarketSegment(("Male", "Old", "HighIncome")),
            MarketSegment(("Female", "Young", "LowIncome")),
            MarketSegment(("Female", "Young", "HighIncome")),
            MarketSegment(("Female", "Old", "LowIncome")),
            MarketSegment(("Female", "Old", "HighIncome"))
        ],
        'market_segment_pop': {
            MarketSegment(("Male", "Young")): 2353,
            MarketSegment(("Male", "Old")): 2603,
            MarketSegment(("Male", "LowIncome")): 3631,
            MarketSegment(("Male", "HighIncome")): 1325,
            MarketSegment(("Female", "Young")): 2236,
            MarketSegment(("Female", "Old")): 2808,
            MarketSegment(("Female", "LowIncome")): 4381,
            MarketSegment(("Female", "HighIncome")): 663,
            MarketSegment(("Young", "LowIncome")): 3816,
            MarketSegment(("Young", "HighIncome")): 773,
            MarketSegment(("Old", "LowIncome")): 4196,
            MarketSegment(("Old", "HighIncome")): 1215,
            MarketSegment(("Male", "Young", "LowIncome")): 1836,
            MarketSegment(("Male", "Young", "HighIncome")): 517,
            MarketSegment(("Male", "Old", "LowIncome")): 1795,
            MarketSegment(("Male", "Old", "HighIncome")): 808,
            MarketSegment(("Female", "Young", "LowIncome")): 1980,
            MarketSegment(("Female", "Young", "HighIncome")): 256,
            MarketSegment(("Female", "Old", "LowIncome")): 2401,
            MarketSegment(("Female", "Old", "HighIncome")): 407
        },
        'user_segment_pmf': {
            MarketSegment(("Male", "Young", "LowIncome")): 0.1836,
            MarketSegment(("Male", "Young", "HighIncome")): 0.0517,
            MarketSegment(("Male", "Old", "LowIncome")): 0.1795,
            MarketSegment(("Male", "Old", "HighIncome")): 0.0808,
            MarketSegment(("Female", "Young", "LowIncome")): 0.1980,
            MarketSegment(("Female", "Young", "HighIncome")): 0.0256,
            MarketSegment(("Female", "Old", "LowIncome")): 0.2401,
            MarketSegment(("Female", "Old", "HighIncome")): 0.0407
        }
    }

def calculate_effective_reach(x: int, R: int) -> float:
    return (2.0 / 4.08577) * (atan(4.08577 * ((x + 0.0) / R) - 3.08577) - atan(-3.08577))

def hash_segment(segment: MarketSegment):

    segments = [
        MarketSegment(("Male", "Young")),
        MarketSegment(("Male", "Old")),
        MarketSegment(("Male", "LowIncome")),
        MarketSegment(("Male", "HighIncome")),
        MarketSegment(("Female", "Young")),
        MarketSegment(("Female", "Old")),
        MarketSegment(("Female", "LowIncome")),
        MarketSegment(("Female", "HighIncome")),
        MarketSegment(("Young", "LowIncome")),
        MarketSegment(("Young", "HighIncome")),
        MarketSegment(("Old", "LowIncome")),
        MarketSegment(("Old", "HighIncome")),
        MarketSegment(("Male", "Young", "LowIncome")),
        MarketSegment(("Male", "Young", "HighIncome")),
        MarketSegment(("Male", "Old", "LowIncome")),
        MarketSegment(("Male", "Old", "HighIncome")),
        MarketSegment(("Female", "Young", "LowIncome")),
        MarketSegment(("Female", "Young", "HighIncome")),
        MarketSegment(("Female", "Old", "LowIncome")),
        MarketSegment(("Female", "Old", "HighIncome"))
    ]
    for i in range(len(segments)):
        if segment == segments[i]: return i
    raise Exception("invalid market segment given to hashing function!")

class MyEnv(gym.Env):

    def __init__(self, env_config):

        # env spaces
        self.action_space = gym.spaces.Box(
            low=np.array([0] * 4, dtype=np.float32),
            high=np.array([1] * 4, dtype=np.float32),
            dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=np.array([-1e4] * (3 + 7*5), dtype=np.float32),
            high=np.array([1e5] * (3 + 7*5), dtype=np.float32),
            dtype=np.float32
        )

        # AdX init
        # if env_config is None:
        #     env_config = CONFIG
        env_config = CONFIG # overwrite config
        self.num_agents = env_config['num_agents']
        self.num_days = env_config['num_days']
        self.α = env_config['quality_score_alpha']
        self.campaigns_per_day = env_config['campaigns_per_day']
        self.agents: List = []
        self.campaign_reach_dist = env_config['campaign_reach_dist']
        self.campaign_length_dist = env_config['campaign_length_dist']
        self.market_segment_dist = env_config['market_segment_dist']
        self.market_segment_pop = env_config['market_segment_pop']
        self.user_segment_dist = PMF(env_config['user_segment_pmf'])
        self.sub_segments = defaultdict(list)
        for user_seg in env_config['user_segment_pmf']:
            for market_seg in env_config['market_segment_dist']:
                if market_seg.issubset(user_seg):
                    self.sub_segments[user_seg].append(market_seg)

    def init_agents(self, agent_types: List[type]) -> Dict[NDaysNCampaignsAgent, CampaignBidderState]:
        states = dict()
        self.agents = []
        for i, agent_type in enumerate(agent_types):
            agent_type.init()
            self.agents.append(agent_type)
            self.agents[i].agent_num = i
            states[agent_type] = CampaignBidderState(i)
        return states 
    
    def generate_campaign(self, start_day: int, end_day: Optional[int] = None) -> Campaign:
        delta = random.choice(self.campaign_reach_dist)
        length =  random.choice(self.campaign_length_dist)
        mkt_segment = random.choice(self.market_segment_dist)
        reach = int(self.market_segment_pop[mkt_segment] * delta)
        if end_day is None:
            end_day=start_day + length - 1
        return Campaign(reach=reach, 
                        target=mkt_segment, 
                        start_day=start_day, 
                        end_day=end_day)
    
    def is_valid_campaign_bid(self, bid: float, reach: int) -> bool:
        return isfinite(bid) and 0.1 * reach <= bid <= reach
        
    def is_valid_bid(self, bid: Bid) -> bool:
        bid = bid.bid_per_item
        return isfinite(bid) and bid > 0

    def run_ad_auctions(self, bid_bundles : List[BidBundle], users: List[MarketSegment], day: int) -> None:
        bidder_states = self.states
        # Map for market_segment to bid
        seg_to_bid = defaultdict(set)
        # Map for campaign_id to limit 
        daily_limits = dict() 
        # Map for bid to bundle 
        bid_to_bundle = dict() 
        # Map from bid entry to spend 
        bid_to_spend = dict()
        for bid_bundle in bid_bundles:
            daily_limits[bid_bundle.campaign_id] = bid_bundle.limit 
            # If campaign does not exist throw-away the bid bundle
            if bid_bundle.campaign_id not in self.campaigns: continue 
            # If campaign is not active throw-away the bid bundle
            campaign = self.campaigns[bid_bundle.campaign_id]
            if not (campaign.start_day <= day and campaign.end_day >= day): continue
            
            for bid in bid_bundle.bid_entries:
                if self.is_valid_bid(bid):
                    bid_to_bundle[bid] = bid_bundle 
                    seg_to_bid[bid.item].add(bid)
                    bid_to_spend[bid] = 0

        for user_segment in users:
            # Get bids that match user
            bids = []
            sub_segments = self.sub_segments[user_segment]
            for seg in sub_segments:
                bids.extend(seg_to_bid[seg])
            
            bids.sort(key=lambda b: b.bid_per_item, reverse=True)

            for i, bid in enumerate(bids):
                campaign_id = bid_to_bundle[bid].campaign_id
                price = bids[i + 1].bid_per_item if i + 1 < len(bids) else 0   
                bidder_states[bid.bidder].spend[campaign_id] += price
                over_bid_limit = bid_to_spend[bid] + price > bid.bid_limit
                over_bundle_limit = bidder_states[bid.bidder].spend[campaign_id] + price > daily_limits[campaign_id]
                if over_bid_limit or over_bundle_limit:
                    # Remove bid if over limit
                    seg_to_bid[bid.item].remove(bid)
                    continue 
                else: 
                    # Update bid 
                    bid_to_spend[bid] += price
                    bidder_state = bidder_states[bid.bidder]
                    campaign_id = bid_to_bundle[bid].campaign_id
                    campaign = bidder_state.campaigns[campaign_id]
                    if campaign is not None:
                        bidder_state.spend[campaign_id] += price
                        campaign.cumulative_cost += price
                        if campaign.target_segment.issubset(user_segment):
                            bidder_state.impressions[campaign_id] += 1
                            campaign.cumulative_reach += 1 
                    break 

    def run_campaign_auctions(self, agent_bids: Dict[NDaysNCampaignsAgent, Dict[Campaign, float]], new_campaigns: List[Campaign]) -> None:
        new_campaigns = set(new_campaigns)
        for campaign in new_campaigns:
            bids = []
            for agent in self.agents:
                if campaign in agent_bids[agent]:
                    reach = campaign.reach
                    agent_bid = agent_bids[agent][campaign]
                    if self.states[agent].quality_score > 0 and self.is_valid_campaign_bid(agent_bid, reach):
                        effective_bid = agent_bid / self.states[agent].quality_score
                        bids.append((agent, effective_bid))
                   
            if bids:
                winner, effective_bid = min(bids, key=lambda x: x[1])
                if len(bids) == 1:
                    q_low = 0.0
                    all_quality_scores = [self.states[agent].quality_score for agent in self.agents]
                    sorted_quality_scores = sorted(all_quality_scores)
                    for i in range(3):
                        q_low += sorted_quality_scores[min(i, len(sorted_quality_scores) - 1)]
                    q_low /= 3
                    budget = campaign.reach / q_low * self.states[winner].quality_score
                else:
                    second_lowest_bid = sorted(bids, key=lambda x: x[1])[1][1]
                    budget = second_lowest_bid * self.states[winner].quality_score
                campaign.budget = budget
                winner.my_campaigns.add(campaign)
                winner_state = self.states[winner]
                winner_state.add_campaign(campaign)
                self.campaigns[campaign.uid] = campaign

    def generate_auction_items(self, num_items: int) -> List[MarketSegment]:
        return [item for item in self.user_segment_dist.draw_n(num_items, replace=True)]

    def reset(self, seed=None, options=None):

        # input agents

        randcamp_agent = RandCampNDaysNCampaignsAgent()
        campaign_agent = CampaignNDaysNCampaignsAgent()
        better_agent = BetterNDaysNCampaignsAgent()
        lessrand_agent = LessRandNDaysNCampaignsAgent(name="LessRand Agent", score_multiplier=-2)
        day_agent = DayNDaysNCampaignsAgent()
        bid_agent = BidNDaysNCampaignsAgent()
        wf_agent = WFNDaysNCampaignsAgent()
        
        test_agents  = [SmartNDaysNCampaignsAgent(name=f"Smart Agent 1", score_multiplier=3), SmartNDaysNCampaignsAgent(name=f"Smart Agent 3", score_multiplier=-3)]
        test_agents += [randcamp_agent, campaign_agent, better_agent, WFNDaysNCampaignsAgent(), day_agent, bid_agent, wf_agent]

        agents: list[NDaysNCampaignsAgent] = [RLNDaysNCampaignsAgent()] + test_agents

        # logging profits
        self.total_profits = {agent : 0.0 for agent in agents}

        # initialize agents
        self.states = self.init_agents(agents)
        self.campaigns = dict()
        # Initialize campaigns 
        for agent in self.agents:    
            # agent.current_game = i + 1 # is this needed for anything?
            agent.my_campaigns = set()
            random_campaign = self.generate_campaign(start_day=1)
            agent_state = self.states[agent]
            random_campaign.budget = random_campaign.reach
            agent_state.add_campaign(random_campaign)
            agent.my_campaigns.add(random_campaign)
            self.campaigns[random_campaign.uid] = random_campaign

        # reset day
        self.day = 0

        # construct obs
        rl_agent = self.agents[0]

        # current day and quality score
        day = self.day
        quality_score = rl_agent.quality_score

        # campaigns
        campaigns = list(rl_agent.get_active_campaigns().union(rl_agent.my_campaigns))
        campaign_features = np.zeros((7*5), dtype=np.float32) # at most 7 campaigns
        for i in range(len(campaigns)):
            campaign_features[0+i*7] = campaigns[i].budget
            campaign_features[1+i*7] = campaigns[i].reach
            campaign_features[2+i*7] = campaigns[i].start
            campaign_features[3+i*7] = campaigns[i].end
            campaign_features[4+i*7] = campaigns[i].cumulative_cost
            campaign_features[5+i*7] = campaigns[i].cumulative_reach
            campaign_features[6+i*7] = hash_segment(campaigns[i].target_segment)

        # final observation
        obs = np.zeros(3 + 7*5)
        obs[0] = day
        obs[1] = quality_score
        obs[2] = rl_agent.profit
        obs[3:] = campaign_features

        # <obs>, <info: dict>
        return obs, {}

    def step_simulation(self) -> bool:

        # update day
        self.day += 1
        simulation_complete = self.day == self.num_days

        # Update 
        for agent in self.agents:
            agent.current_day = self.day

        # Generate new campaigns and filter
        if self.day + 1 < self.num_days + 1:
            self.new_campaigns = [self.generate_campaign(start_day=self.day + 1) for _ in range(self.campaigns_per_day)]
            self.new_campaigns = [c for c in self.new_campaigns if c.end_day <= self.num_days]
            
            # Solicit campaign bids and run campaign auctions
            self.agent_bids = dict()
            for agent in self.agents:
                self.agent_bids[agent] = agent.get_campaign_bids(self.new_campaigns)

        # Solicit ad bids from agents and run ad auctions
        ad_bids = []
        for agent in self.agents:
            ad_bids.extend(agent.get_ad_bids())
        users = self.generate_auction_items(10000)
        self.run_ad_auctions(ad_bids, users, self.day)

        # Update campaign states, quality scores, and profits
        for agent in self.agents:
            agent_state = self.states[agent]
            todays_profit = 0.0
            new_qs_count = 0
            new_qs_val = 0.0

            for campaign in agent_state.campaigns.values():
                if campaign.start_day <= self.day <= campaign.end_day:
                    if self.day == campaign.end_day:
                        impressions = agent_state.impressions[campaign.uid]
                        total_cost = agent_state.spend[campaign.uid]
                        effective_reach = calculate_effective_reach(impressions, campaign.reach)
                        todays_profit += (effective_reach) * agent_state.budgets[campaign.uid] - total_cost

                        new_qs_count += 1
                        new_qs_val += effective_reach

            if new_qs_count > 0:
                new_qs_val /= new_qs_count
                self.states[agent].quality_score = (1 - self.α) * self.states[agent].quality_score + self.α * new_qs_val
                agent.quality_score = self.states[agent].quality_score

            agent_state.profits += todays_profit
            agent.profit += todays_profit
        
        # Run campaign auctions
        self.run_campaign_auctions(self.agent_bids, self.new_campaigns)
        # Run campaign endowments
        for agent in self.agents:
            if random.random() < min(1, agent.quality_score):
                random_campaign = self.generate_campaign(start_day=self.day)
                agent_state = self.states[agent]
                random_campaign.budget = random_campaign.reach
                agent_state.add_campaign(random_campaign)
                agent.my_campaigns.add(random_campaign)
                self.campaigns[random_campaign.uid] = random_campaign

        # logging total profits
        for agent in self.agents:
            self.total_profits[agent] += self.states[agent].profits 

        # stop after max days reached
        return simulation_complete

    def step(self, action):

        # cache rl-agent
        rl_agent: NDaysNCampaignsAgent = self.agents[0]

        # action







        
        # redefine agent bidding functions
        def rl_agent_get_ad_bids() -> Set[BidBundle]:
            bundles = set()
            campaigns = rl_agent.get_active_campaigns().union(rl_agent.my_campaigns)

            # use waterfall algorithm
            allocations = waterfall(rl_agent.current_day, campaigns.union(rl_agent.campaign_store))

            # find the lowest price among our allocations
            prices = {}
            for campaign in campaigns:
                if campaign not in allocations: continue
                for segment, price, allocation in allocations[campaign]:
                    if segment in prices:
                        prices[segment] = min(prices[segment], price)
                    else:
                        prices[segment] = price

            # bids
            for campaign in campaigns:

                # checking if the campaign is active
                if campaign.end_day < rl_agent.current_day: continue
                if campaign.start_day > rl_agent.current_day: continue

                # bidding calculations
                bid_entries = set()
                subsets = set()
                for segment in market_segments:
                    if campaign.target_segment.issuperset(segment):
                        subsets.add(segment)
                n = len(subsets)
                for segment in subsets:
                    auction_item = segment
                    bid_per_item = campaign.budget / (n * campaign.reach)

                    # use waterfall bidding
                    if segment in prices:
                        bid_per_item = action[0] * bid_per_item + (1 - action[0]) * prices[segment]
                        # bid_per_item = (bid_per_item + prices[segment]) / 2

                    bid = Bid(
                        bidder=rl_agent,
                        auction_item=auction_item,
                        bid_per_item=bid_per_item,
                        bid_limit=campaign.budget / n
                    )
                    bid_entries.add(bid)
                limit = campaign.budget
                bundle = BidBundle(
                    campaign_id=campaign.uid,
                    limit=limit,
                    bid_entries=bid_entries
                )
                bundles.add(bundle)
            return bundles
    

        def rl_agent_get_campaign_bids(campaigns_for_auction: Set[Campaign]) -> Dict[Campaign, float]:
            # rl_agent.count_won_auctions(campaigns_for_auction)
            # bids = {}
            # for campaign in campaigns_for_auction:
            #     # bids[campaign] = campaign.reach * 0.125
            #     bids[campaign] = campaign.reach * 0.1
            # return bids

            rl_agent.campaign_store = rl_agent.campaign_store.union(campaigns_for_auction)

            # parameters
            delta = 0.1
            min_alpha = 0.1
            # reward_multiplier = 0.25
            # score_multiplier = -3.0
            # day_multiplier = 1.

            reward_multiplier = 0.25 * (1. + action[1] - 0.5)
            score_multiplier = (action[2] - 0.5) * 6
            day_multiplier = (1. + action[3] - 0.5)

            # take own campaigns into account
            my_campaigns = rl_agent.get_active_campaigns().union(rl_agent.my_campaigns)

            # remove inactive campaigns
            reduced_campaigns: Set[Campaign] = set()
            for campaign in my_campaigns:
                if campaign.end_day < rl_agent.current_day:
                    continue
                reduced_campaigns.add(campaign)

            # update auction prices for each campaign
            for campaign in rl_agent.campaign_auction_history:
                if campaign not in my_campaigns:
                    alpha = rl_agent.campaign_history[campaign.target_segment] - delta

                    # alpha *= 1. + (self.random.random() - 0.45) / 5 # randomness
                    # alpha *= 1. + (self.random.random() - 0.45) / 8 # randomness

                    # take into account the number of days
                    day_difference = float(campaign.end_day - campaign.start_day) / 6.
                    if day_difference == 0.:
                        day_difference = 1.
                    # alpha *= 1 + day_difference
                    alpha *= (1 - day_difference) * day_multiplier

                    if len(reduced_campaigns) > 0: 
                        score = 0
                        for reduced_campaign in reduced_campaigns:
                            if reduced_campaign.target_segment.issubset(campaign.target_segment) or reduced_campaign.target_segment.issuperset(campaign.target_segment):
                                score += 1
                        score /= len(reduced_campaigns)
                        alpha *= 1. - score * score_multiplier
                    
                    rl_agent.campaign_history[campaign.target_segment] = max(alpha, min_alpha)
                else:
                    rl_agent.campaign_history[campaign.target_segment] += delta * reward_multiplier

            # counting for stats
            rl_agent.count_won_auctions(campaigns_for_auction)

            # create bids
            bids = {}
            for campaign in campaigns_for_auction:
                if campaign.target_segment not in rl_agent.campaign_history:
                    rl_agent.campaign_history[campaign.target_segment] = 0.25
                bids[campaign] = campaign.reach * rl_agent.campaign_history[campaign.target_segment]
                # bids[campaign] *= 1. + (self.random.random() - 0.45) / 3

            # return campaign bids
            return bids
        









        # redefine agent bidding functions
        rl_agent.get_ad_bids = rl_agent_get_ad_bids
        rl_agent.get_campaign_bids = rl_agent_get_campaign_bids
        
        # step AdX auction
        complete = self.step_simulation()

        # current day and quality score
        day = self.day
        quality_score = rl_agent.quality_score

        # campaigns
        campaigns = list(rl_agent.get_active_campaigns().union(rl_agent.my_campaigns))
        campaign_features = np.zeros((7*5), dtype=np.float32) # at most 5 campaigns
        for i in range(len(campaigns)):
            if i >= 5: break # capping out
            campaign_features[0+i*7] = campaigns[i].budget
            campaign_features[1+i*7] = campaigns[i].reach
            campaign_features[2+i*7] = campaigns[i].start
            campaign_features[3+i*7] = campaigns[i].end
            campaign_features[4+i*7] = campaigns[i].cumulative_cost
            campaign_features[5+i*7] = campaigns[i].cumulative_reach
            campaign_features[6+i*7] = hash_segment(campaigns[i].target_segment)

        # final observation
        obs = np.zeros(3 + 7*5)
        obs[0] = day
        obs[1] = quality_score
        obs[2] = rl_agent.profit / 30000
        obs[3:] = campaign_features



        # <obs>, <reward: float>, <terminated: bool>, <truncated: bool>, <info: dict>
        return obs, rl_agent.profit / 30000, complete, False, {}
    
# env testing
if __name__ == "__main__":
    print("Hello Env!")

    env = MyEnv(None)
    env.reset()

    for _ in range(5):  
        obs, rewards, done, truncated, info = env.step(env.action_space.sample())
        print(obs)