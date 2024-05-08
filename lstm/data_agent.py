from typing import Set, Dict, List
import numpy as np
import sys, os
sys.path.append("..")
import random
import json

from adx.agents import NDaysNCampaignsAgent
from adx.tier1_ndays_ncampaign_agent import Tier1NDaysNCampaignsAgent
# from adx.adx_game_simulator import AdXGameSimulator
from adx_game_simulator_truth import AdXGameSimulator
from adx.structures import Bid, Campaign, BidBundle, MarketSegment
from adx.adx_game_simulator import calculate_effective_reach

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

        self.scale = np.ones(26).tolist()

        self.open_campaigns: Set[Campaign] = set()  # hold all open campaigns
        self.history = []
        self.scale = np.ones(26).tolist() 

        self.market_dict: Dict[MarketSegment, int] = {ms: idx for idx, ms in enumerate(market_segments)}
        self.target_dict: Dict[MarketSegment, int] = {ts: idx for idx, ts in enumerate(target_segments)}

        self.game_features_list: List[List[float]] = []
        self.features_list: List[List[List[float]]] = []
    
    def on_new_game(self) -> None:
        pass

    def get_ad_bids(self) -> Set[BidBundle]:

        # make feature vectors...
        features = self.scale
        # print(features)
        self.game_features_list.append(features)
        # print(self.game_features_list)

        if self.current_day == 10:
            self.features_list.append(self.game_features_list)
            self.game_features_list = []
      

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
        # my_campaigns = self.get_active_campaigns().union(self.my_campaigns)
        # self.open_campaigns = self.open_campaigns.union(campaigns_for_auction)
        # self.open_campaigns = self.open_campaigns.union(my_campaigns)

        # temporary bidding
        bids = {}
        for campaign in campaigns_for_auction:
            bids[campaign] = campaign.reach * 2
            lr = 0.5
            bid = bids[campaign] / campaign.reach
            self.scale[self.target_dict[campaign.target_segment]] = (1 - lr) * self.scale[self.target_dict[campaign.target_segment]] + lr * bid
        
        return bids
    

if __name__ == "__main__":
    # Here's an opportunity to test offline against some TA agents. Just run this file to do so.
    data_agent = DataNDaysNCampaignsAgent()
    test_agents = [data_agent] + [Tier1NDaysNCampaignsAgent(name=f"Agent {i + 1}") for i in range(9)]

    # Don't change this. Adapt initialization to your environment
    simulator = AdXGameSimulator()
    campaign_history: List[List[Set[Campaign]]] = simulator.run_simulation(agents=test_agents, num_simulations=10) # originally 500 

    # json dump or something
    # data_agent.features_list # json dump or something here..

    # store simulator data somewhow....

    json_campaign_history = []
    for item in campaign_history:
        outer_list = []
        for inner_item in item:
            inner_dict = {}
            for segment in target_segments:
                inner_dict[segment] = 0
            for campaign in list(inner_item):
                inner_dict[campaign.target_segment] += 1
            sum = np.sum([count for count in inner_dict.values()])
            inner_list = [count / sum for count in inner_dict.values()]
            outer_list.append(inner_list)
        json_campaign_history.append(outer_list)
    

    # print(json_campaign_history)
    # print(data_agent.features_list)

    out_dict = {
        "labels": json_campaign_history,
        "features": data_agent.features_list
    }

    out_file = open("data.json", "w")
    json.dump(out_dict, out_file) 
    out_file.close() 





