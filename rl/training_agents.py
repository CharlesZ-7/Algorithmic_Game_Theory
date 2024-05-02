
from typing import Set, Dict, List

import sys, os
sys.path.append("..")
import random

from adx.agents import NDaysNCampaignsAgent
from adx.tier1_ndays_ncampaign_agent import Tier1NDaysNCampaignsAgent
from adx.adx_game_simulator import AdXGameSimulator
from adx.structures import Bid, Campaign, BidBundle, MarketSegment
from waterfall_algorithm import waterfall


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


class CriticalNDaysNCampaignsAgent(NDaysNCampaignsAgent):

    def __init__(self, name = "Critical Agent"):
        super().__init__()
        self.name = name

    def on_new_game(self) -> None:
        pass

    def get_ad_bids(self) -> Set[BidBundle]:
        bundles = set()
        campaigns = self.get_active_campaigns().union(self.my_campaigns)
        for campaign in campaigns:
            bid_entries = set()
            subsets = set()
            # for segment in MarketSegment.all_segments():
            for segment in market_segments:
                if campaign.target_segment.issuperset(segment):
                    subsets.add(segment)
            n = len(subsets)
            for segment in subsets:
                auction_item = segment
                bid_per_item = campaign.budget / (n * campaign.reach)
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
        bids = {}
        for campaign in campaigns_for_auction:
            bids[campaign] = campaign.reach * 2
        return bids


class RandomNDaysNCampaignsAgent(NDaysNCampaignsAgent):

    def __init__(self, name = "Random Agent"):
        super().__init__()
        self.name = name
        self.random = random

    def on_new_game(self) -> None:
        pass

    def get_ad_bids(self) -> Set[BidBundle]:
        bundles = set()
        campaigns = self.get_active_campaigns().union(self.my_campaigns)
        for campaign in campaigns:
            bid_entries = set()
            subsets = set()
            for segment in market_segments:
                if campaign.target_segment.issuperset(segment):
                    subsets.add(segment)
            n = len(subsets)
            for segment in subsets:
                auction_item = segment
                bid_per_item = campaign.budget / (n * campaign.reach)
                alpha = 1.0 + (self.random.random() - 0.5) / 3.0
                bid_per_item *= alpha
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
        bids = {}
        for campaign in campaigns_for_auction:
            bids[campaign] = campaign.reach * self.random.random()
        return bids


class BetterNDaysNCampaignsAgent(CriticalNDaysNCampaignsAgent):

    def __init__(self, name = "Better Agent"):
        super().__init__()
        self.name = name

        self.auctions = 0
        self.auctions_won = 0

        self.campaign_auction_history: List[Campaign] = []
        
    
    def count_won_auctions(self, campaigns_for_auction:  Set[Campaign]):
        
        # count number of won auctions
        self.auctions += len(self.campaign_auction_history)
        my_campaigns = self.get_active_campaigns().union(self.my_campaigns)
        for campaign in self.campaign_auction_history:
            if campaign in my_campaigns:
                self.auctions_won += 1

        # store campaigns for next iteration
        self.campaign_auction_history = campaigns_for_auction

    def get_campaign_bids(self, campaigns_for_auction:  Set[Campaign]) -> Dict[Campaign, float]:
        self.count_won_auctions(campaigns_for_auction)
        bids = {}
        for campaign in campaigns_for_auction:
            # bids[campaign] = campaign.reach * 0.125
            bids[campaign] = campaign.reach * 0.1
        return bids
    

class CampaignNDaysNCampaignsAgent(BetterNDaysNCampaignsAgent):

    def __init__(self, name = "Campaign Agent"):
        super().__init__()
        self.name = name

        self.random = random
        self.campaign_history: Dict[MarketSegment, float] = {}

    def on_new_game(self) -> None:
        # self.campaign_history = {} # reset history... does not improve performance...
        pass

    def get_campaign_bids(self, campaigns_for_auction:  Set[Campaign]) -> Dict[Campaign, float]:

        # parameters
        delta = 0.1
        min_alpha = 0.1
        reward_multiplier = 0.25

        # update auction prices for each campaign
        for campaign in self.campaign_auction_history:
            my_campaigns = self.get_active_campaigns().union(self.my_campaigns)
            # if self.random.random() >= 0.5:
            if campaign not in my_campaigns:
                alpha = self.campaign_history[campaign.target_segment] - delta
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

        # return campaign bids
        return bids
    

class RandCampNDaysNCampaignsAgent(BetterNDaysNCampaignsAgent):

    def __init__(self, name = "RandCamp Agent"):
        super().__init__()
        self.name = name

        self.random = random
        self.campaign_history: Dict[MarketSegment, float] = {}

    def on_new_game(self) -> None:
        # self.campaign_history = {} # reset history... does not improve performance...
        pass

    def get_campaign_bids(self, campaigns_for_auction:  Set[Campaign]) -> Dict[Campaign, float]:

        # parameters
        delta = 0.1
        min_alpha = 0.1
        reward_multiplier = 0.25

        # update auction prices for each campaign
        for campaign in self.campaign_auction_history:
            my_campaigns = self.get_active_campaigns().union(self.my_campaigns)
            if campaign not in my_campaigns:
                alpha = self.campaign_history[campaign.target_segment] - delta
                alpha *= 1. + (self.random.random() - 0.45) / 3 # randomness
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
    

class SmartNDaysNCampaignsAgent(BetterNDaysNCampaignsAgent):

    def __init__(self, name = "Smart Agent", score_multiplier: float = 1.):
        super().__init__()
        self.name = name

        self.score_multiplier = score_multiplier
        self.random = random
        self.campaign_history: Dict[MarketSegment, float] = {}

    def on_new_game(self) -> None:
        # self.campaign_history = {} # reset history... does not improve performance...
        pass

    def get_campaign_bids(self, campaigns_for_auction:  Set[Campaign]) -> Dict[Campaign, float]:

        # parameters
        delta = 0.1
        min_alpha = 0.1
        reward_multiplier = 0.25

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

                alpha *= 1. + (self.random.random() - 0.45) / 3 # randomness

                if len(reduced_campaigns) > 0: 
                    score = 0
                    for reduced_campaign in reduced_campaigns:
                        if reduced_campaign.target_segment.issubset(campaign.target_segment) or reduced_campaign.target_segment.issuperset(campaign.target_segment):
                            score += 1
                    score /= len(reduced_campaigns)
                    alpha *= 1. - score * self.score_multiplier
                    # alpha *= 1. + score * 0.1
                
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
    
class LessRandNDaysNCampaignsAgent(BetterNDaysNCampaignsAgent):

    def __init__(self, name = "LessRand Agent", score_multiplier: float = 1.):
        super().__init__()
        self.name = name

        self.score_multiplier = score_multiplier
        self.random = random
        self.campaign_history: Dict[MarketSegment, float] = {}

    def on_new_game(self) -> None:
        # self.campaign_history = {} # reset history... does not improve performance...
        pass

    def get_campaign_bids(self, campaigns_for_auction:  Set[Campaign]) -> Dict[Campaign, float]:

        # parameters
        delta = 0.1
        min_alpha = 0.1
        reward_multiplier = 0.25

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

                alpha *= 1. + (self.random.random() - 0.45) / 5 # randomness
                # alpha *= 1. + (self.random.random() - 0.45) / 8 # randomness

                # take into account the number of days...

                if len(reduced_campaigns) > 0: 
                    score = 0
                    for reduced_campaign in reduced_campaigns:
                        if reduced_campaign.target_segment.issubset(campaign.target_segment) or reduced_campaign.target_segment.issuperset(campaign.target_segment):
                            score += 1
                    score /= len(reduced_campaigns)
                    alpha *= 1. - score * self.score_multiplier
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


class DayNDaysNCampaignsAgent(BetterNDaysNCampaignsAgent):

    def __init__(self, name = "Day Agent", score_multiplier: float = 1.):
        super().__init__()
        self.name = name

        self.score_multiplier = score_multiplier
        self.random = random
        self.campaign_history: Dict[MarketSegment, float] = {}

    def on_new_game(self) -> None:
        # self.campaign_history = {} # reset history... does not improve performance...
        pass

    def get_campaign_bids(self, campaigns_for_auction:  Set[Campaign]) -> Dict[Campaign, float]:

        # parameters
        delta = 0.1
        min_alpha = 0.1
        reward_multiplier = 0.25

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
                alpha *= 1 - day_difference

                if len(reduced_campaigns) > 0: 
                    score = 0
                    for reduced_campaign in reduced_campaigns:
                        if reduced_campaign.target_segment.issubset(campaign.target_segment) or reduced_campaign.target_segment.issuperset(campaign.target_segment):
                            score += 1
                    score /= len(reduced_campaigns)
                    alpha *= 1. - score * self.score_multiplier
                
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


class BidNDaysNCampaignsAgent(BetterNDaysNCampaignsAgent):

    def __init__(self, name = "Bid Agent"):
        super().__init__()
        self.name = name

    def on_new_game(self) -> None:
        pass

    def get_ad_bids(self) -> Set[BidBundle]:
        bundles = set()
        campaigns = self.get_active_campaigns().union(self.my_campaigns)
        for campaign in campaigns:
            bid_entries = set()
            subsets = set()
            # for segment in MarketSegment.all_segments():
            for segment in market_segments:
                if campaign.target_segment.issuperset(segment):
                    subsets.add(segment)
            n = len(subsets)
            for segment in subsets:
                auction_item = segment
                bid_per_item = campaign.budget / (n * campaign.reach)
                bid = Bid(
                    bidder=self,
                    auction_item=auction_item,
                    bid_per_item=bid_per_item * 1. - .0125,
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
    

class WFNDaysNCampaignsAgent(BetterNDaysNCampaignsAgent):

    def __init__(self, name = "WF Agent"):
        super().__init__()
        self.name = name

        self.campaign_store = set()

    def on_new_game(self) -> None:
        pass

    def get_ad_bids(self) -> Set[BidBundle]:
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
                    bid_per_item = (bid_per_item + prices[segment]) / 2

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
        self.campaign_store = self.campaign_store.union(campaigns_for_auction)
        return super().get_campaign_bids(campaigns_for_auction)


if __name__ == "__main__":
    # Here's an opportunity to test offline against some TA agents. Just run this file to do so.
    # test_agents = [CriticalNDaysNCampaignsAgent()] + [Tier1NDaysNCampaignsAgent(name=f"Agent {i + 1}") for i in range(9)]
    
    # smart_agent = SmartNDaysNCampaignsAgent()
    randcamp_agent = RandCampNDaysNCampaignsAgent()
    campaign_agent = CampaignNDaysNCampaignsAgent()
    better_agent = BetterNDaysNCampaignsAgent()
    lessrand_agent = LessRandNDaysNCampaignsAgent(name="LessRand Agent", score_multiplier=-2)
    day_agent = DayNDaysNCampaignsAgent()
    bid_agent = BidNDaysNCampaignsAgent()
    wf_agent = WFNDaysNCampaignsAgent()
    
    test_agents  = [SmartNDaysNCampaignsAgent(name=f"Smart Agent 1", score_multiplier=3), SmartNDaysNCampaignsAgent(name=f"Smart Agent 2", score_multiplier=-2), SmartNDaysNCampaignsAgent(name=f"Smart Agent 3", score_multiplier=-3)]
    test_agents += [randcamp_agent, campaign_agent, better_agent, lessrand_agent, day_agent, bid_agent, wf_agent]
    # test_agents += [RandomNDaysNCampaignsAgent(name=f"Rng Agent {i + 1}") for i in range(1)]
    # test_agents += [CriticalNDaysNCampaignsAgent(name=f"Crit Agent {i + 1}") for i in range(1)]

    # Don't change this. Adapt initialization to your environment
    simulator = AdXGameSimulator()
    simulator.run_simulation(agents=test_agents, num_simulations=500)

    # agent history debugging
    def stats_print(agent: BetterNDaysNCampaignsAgent):
        print(f"\n{agent.name} won {agent.auctions_won}/{agent.auctions} auctions | {round(float(agent.auctions_won) / float(agent.auctions) * 100, 3)}%")
    
    for agent in test_agents:
        if hasattr(agent, 'auctions_won'):
            stats_print(agent)
        else:
            print(f"{agent.name} no stats!")