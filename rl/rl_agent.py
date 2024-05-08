from typing import Set

import sys, os
sys.path.append("..")
from adx.structures import BidBundle
from training_agents import *
from ray.rllib.algorithms import Algorithm
import numpy as np
from env import hash_segment

from env import MyEnv

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

class RLModelNDaysNCampaignsAgent(BetterNDaysNCampaignsAgent):

    def __init__(self, name = "RL Agent"):
        super().__init__()
        self.name = name

        # history variables
        self.campaign_store = set()
        self.campaign_history: Dict[MarketSegment, float] = {}

        # rl model
        self.model = Algorithm.from_checkpoint(os.path.join(os.getcwd(), "checkpoints"))
    
    def get_action(self):
        # campaigns
        campaigns = list(self.get_active_campaigns().union(self.my_campaigns))
        reduced_campaigns = []
        for campaign in campaigns:
            if campaign.end_day < self.current_day: continue
            if campaign.start_day > self.current_day: continue
            reduced_campaigns.append(campaign)
        campaigns = reduced_campaigns
        campaign_features = np.zeros((7*5), dtype=np.float32) # at most 7 campaigns


        # maybe i should pick out better which campaigns get selected...


        for i in range(len(campaigns)):
            if i == 5: break
            campaign_features[0+i*7] = campaigns[i].budget
            campaign_features[1+i*7] = campaigns[i].reach
            campaign_features[2+i*7] = campaigns[i].start
            campaign_features[3+i*7] = campaigns[i].end
            campaign_features[4+i*7] = campaigns[i].cumulative_cost
            campaign_features[5+i*7] = campaigns[i].cumulative_reach
            campaign_features[6+i*7] = hash_segment(campaigns[i].target_segment)

        # final observation
        obs = np.zeros(3 + 7*5)
        obs[0] = self.current_day
        obs[1] = self.quality_score
        obs[2] = self.profit
        obs[3:] = campaign_features

        # get and store action
        return self.model.compute_single_action(obs)

    def get_ad_bids(self) -> Set[BidBundle]:

        # get rl action
        action = self.get_action()



        # bidding calculation 
        bundles = set()
        campaigns = self.get_active_campaigns().union(self.my_campaigns)

        # use waterfall algorithm
        allocations = waterfall(self.current_day, campaigns.union(self.campaign_store))

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
            if campaign.end_day < self.current_day: continue
            if campaign.start_day > self.current_day: continue

            # checking reach and budget
            reach_value = campaign.reach - campaign.cumulative_reach
            budget_value = campaign.budget - campaign.cumulative_cost
            if reach_value <= 0 or budget_value <= 0: continue

            # bidding calculations
            bid_entries = set()
            subsets = set()
            for segment in market_segments:
                if campaign.target_segment.issuperset(segment):
                    subsets.add(segment)
            n = len(subsets)
            for segment in subsets:
                auction_item = segment
                # bid_per_item = campaign.budget / (n * campaign.reach)
                bid_per_item = budget_value / (n * reach_value)

                # use waterfall bidding
                if segment in prices:
                    bid_per_item = action[0] * bid_per_item + (1 - action[0]) * prices[segment]
                    # bid_per_item = (bid_per_item + prices[segment]) / 2

                bid = Bid(
                    bidder=self,
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

    def get_campaign_bids(self, campaigns_for_auction:  Set[Campaign]) -> Dict[Campaign, float]:

        self.log_stats(campaigns_for_auction)

        # get rl action
        action = self.get_action()

        # self.count_won_auctions(campaigns_for_auction)
        # bids = {}
        # for campaign in campaigns_for_auction:
        #     # bids[campaign] = campaign.reach * 0.125
        #     bids[campaign] = campaign.reach * 0.1
        # return bids

        self.campaign_store = self.campaign_store.union(campaigns_for_auction)

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
        my_campaigns = self.get_active_campaigns().union(self.my_campaigns)

        # remove inactive campaigns
        reduced_campaigns: Set[Campaign] = set()
        for campaign in my_campaigns:
            if campaign.end_day < self.current_day:
                continue
            reduced_campaigns.add(campaign)

        # update auction prices for each campaign
        for campaign in self.campaign_auction_history:
            if campaign not in my_campaigns:
                alpha = self.campaign_history[campaign.target_segment] - delta

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
                
                self.campaign_history[campaign.target_segment] = max(alpha, min_alpha)
            else:
                self.campaign_history[campaign.target_segment] += delta * reward_multiplier

        # counting for stats
        self.count_won_auctions(campaigns_for_auction)

        # create bids
        bids = {}
        for campaign in campaigns_for_auction:
            if campaign.target_segment not in self.campaign_history:
                self.campaign_history[campaign.target_segment] = 0.25
            bids[campaign] = campaign.reach * self.campaign_history[campaign.target_segment]
            # bids[campaign] *= 1. + (self.random.random() - 0.45) / 3

        # return campaign bids
        return bids


if __name__ == "__main__":

    print("hello world!")


    # add custom env to registry
    def env_creator(env_config):
        return MyEnv(env_config)
    register_env("my_env", env_creator)

    # Here's an opportunity to test offline against some TA agents. Just run this file to do so.
    # test_agents = [RLModelNDaysNCampaignsAgent()] + [Tier1NDaysNCampaignsAgent(name=f"Agent {i + 1}") for i in range(9)]
    # simulator = AdXGameSimulator()
    # simulator.run_simulation(agents=test_agents, num_simulations=250)







    # smart_agent = SmartNDaysNCampaignsAgent()
    randcamp_agent = RandCampNDaysNCampaignsAgent()
    campaign_agent = CampaignNDaysNCampaignsAgent()
    better_agent = BetterNDaysNCampaignsAgent()
    lessrand_agent = LessRandNDaysNCampaignsAgent(name="LessRand Agent", score_multiplier=-2)
    day_agent = DayNDaysNCampaignsAgent()
    bid_agent = BidNDaysNCampaignsAgent()
    wf_agent = WFNDaysNCampaignsAgent()
    
    test_agents  = [SmartNDaysNCampaignsAgent(name=f"Smart Agent 1", score_multiplier=3), SmartNDaysNCampaignsAgent(name=f"Smart Agent 2", score_multiplier=-2), SmartNDaysNCampaignsAgent(name=f"Smart Agent 3", score_multiplier=-3)]
    test_agents += [RLModelNDaysNCampaignsAgent(), campaign_agent, better_agent, lessrand_agent, day_agent, bid_agent, wf_agent]
    # test_agents += [RandomNDaysNCampaignsAgent(name=f"Rng Agent {i + 1}") for i in range(1)]
    # test_agents += [CriticalNDaysNCampaignsAgent(name=f"Crit Agent {i + 1}") for i in range(1)]

    # Don't change this. Adapt initialization to your environment
    simulator = AdXGameSimulator()
    simulator.run_simulation(agents=test_agents, num_simulations=250)




    # # agent history debugging
    # def stats_print(agent: BetterNDaysNCampaignsAgent):
    #     print(f"\n{agent.name} won {agent.auctions_won}/{agent.auctions} auctions | {round(float(agent.auctions_won) / float(agent.auctions) * 100, 3)}%")
    
    # for agent in test_agents:
    #     if hasattr(agent, 'auctions_won'):
    #         stats_print(agent)
    #     else:
    #         print(f"{agent.name} no stats!")


    log = {}
    for agent in test_agents:
        if not hasattr(agent, 'auctions_won'): continue

        stats = {
            "total_campaigns_won": agent.auctions_won,
            "profits": agent.profit_history,
            "campaign_auction_winnings": agent.campaign_winning_history,
            "quality_scores": agent.quality_score_history
        }

        log[agent.name] = stats


    # print(log)

    with open("data.json", "w") as f:
        json.dump(log, f) 