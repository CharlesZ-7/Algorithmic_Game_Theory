from typing import Set, Dict, List

import sys, os
sys.path.append("..")
import random

from adx.agents import NDaysNCampaignsAgent
from adx.tier1_ndays_ncampaign_agent import Tier1NDaysNCampaignsAgent
# from adx.adx_game_simulator import AdXGameSimulator
from adx_game_simulator_truth import AdXGameSimulator
from adx.structures import Bid, Campaign, BidBundle, MarketSegment

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

target_segments = [
    MarketSegment(("Male", "Young", "LowIncome")),
    MarketSegment(("Male", "Young", "HighIncome")),
    MarketSegment(("Male", "Old", "LowIncome")),
    MarketSegment(("Male", "Old", "HighIncome")),
    MarketSegment(("Female", "Young", "LowIncome")),
    MarketSegment(("Female", "Young", "HighIncome")),
    MarketSegment(("Female", "Old", "LowIncome")),
    MarketSegment(("Female", "Old", "HighIncome")),
    MarketSegment(("Male", "Young")),
    MarketSegment(("Male", "Old",)),
    MarketSegment(("Male", "HighIncome")),
    MarketSegment(("Male", "LowIncome")),
    MarketSegment(("Female", "Young")),
    MarketSegment(("Female", "Old",)),
    MarketSegment(("Female", "HighIncome")),
    MarketSegment(("Female", "LowIncome")),
    MarketSegment(("Young", "LowIncome")),
    MarketSegment(("Young", "HighIncome")),
    MarketSegment(("Old", "LowIncome")),
    MarketSegment(("Old", "HighIncome")),
    MarketSegment(("Male",)),
    MarketSegment(("Female",)),
    MarketSegment(("Young",)),
    MarketSegment(("Old",)),
    MarketSegment(("HighIncome",)),
    MarketSegment(("LowIncome",))
]

class DataNDaysNCampaignsAgent(NDaysNCampaignsAgent):

    def __init__(self, name = "Data Agent"):
        super().__init__()
        self.name = name

        self.all_campaigns: Set[Campaign] = set() # hold all campaigns

        # history value
        self.history = []
    
    def on_new_game(self) -> None:
        pass

    def get_ad_bids(self) -> Set[BidBundle]:

        # make feature vectors...
            # TODO

        # get active campaigns
        active_campaigns: Set[Campaign] = set()
        for campaign in list(self.all_campaigns):
            if campaign.end_day < self.current_day: continue
            if campaign.start_day > self.current_day: continue
            active_campaigns.add(campaign)


        # make open_campaigns_list

        open_campaigns_list = np.zeros(26)

        for index in range(26):
            for campaign in active_campaigns:
                if target_segment[index] == campaign.target_segment:
                    open_campaigns_list[index] += 1
            

        # store this somewhere...




        # leave this be...
        
        # critical bidding
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
        
        # store campaigns
        my_campaigns = self.get_active_campaigns().union(self.my_campaigns)
        self.all_campaigns = self.all_campaigns.union(campaigns_for_auction)
        self.all_campaigns = self.all_campaigns.union(my_campaigns)

        # temporary bidding
        bids = {}
        for campaign in campaigns_for_auction:
            bids[campaign] = campaign.reach * 2
        return bids
    

if __name__ == "__main__":
    # Here's an opportunity to test offline against some TA agents. Just run this file to do so.
    data_agent = DataNDaysNCampaignsAgent()
    test_agents = [data_agent] + [Tier1NDaysNCampaignsAgent(name=f"Agent {i + 1}") for i in range(9)]

    # Don't change this. Adapt initialization to your environment
    simulator = AdXGameSimulator()
    campaign_history: List[List[Set[Campaign]]] = simulator.run_simulation(agents=test_agents, num_simulations=500) # originally 500 

    # json dump or something
    data_agent.history # json dump or something here..

    # store simulator data somewhow....
