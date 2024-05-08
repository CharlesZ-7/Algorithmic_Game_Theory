from typing import Set, Dict, List

import sys, os
sys.path.append("..")
import random
import numpy as np

import json
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset, random_split

from adx.agents import NDaysNCampaignsAgent
from adx.tier1_ndays_ncampaign_agent import Tier1NDaysNCampaignsAgent
from adx.adx_game_simulator import AdXGameSimulator
from adx.structures import Bid, Campaign, BidBundle, MarketSegment

from model import CampaignPredictor, load_checkpoint
from solver import User, Solution

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

class LSTMNDaysNCampaignsAgent(NDaysNCampaignsAgent):

    def __init__(self, name = "LSTM Agent"):
        super().__init__()
        self.name = name

        self.open_campaigns: Set[Campaign] = set() # hold open campaigns

        self.users = set()

        self.users.add(User(MarketSegment(("Male", "Young", "LowIncome")), 1836))
        self.users.add(User(MarketSegment(("Male", "Young", "HighIncome")), 517))
        self.users.add(User(MarketSegment(("Male", "Old", "LowIncome")), 1795))
        self.users.add(User(MarketSegment(("Male", "Old", "HighIncome")), 808))
        self.users.add(User(MarketSegment(("Female", "Young", "LowIncome")), 1980))
        self.users.add(User(MarketSegment(("Female", "Young", "HighIncome")), 256))
        self.users.add(User(MarketSegment(("Female", "Old", "LowIncome")), 2401))
        self.users.add(User(MarketSegment(("Female", "Old", "HighIncome")), 40))

        # import model and load weights     
        self.model = CampaignPredictor(input_size=26, hidden_size=100, num_layers=3, output_size=26)


        checkpoint_path = 'checkpoint.pth'
        if os.path.exists(checkpoint_path):
            # Load model weights
            _, _ = load_checkpoint(self.model, checkpoint_path) 
        else:
            print("Checkpoint file not found, starting with untrained model.")

        solution = Solution()
        
        self.scale = np.ones(26).tolist()

        self.open_campaigns: Set[Campaign] = set()  # hold all open campaigns
        self.history = []
        self.scale = np.ones(26).tolist() 

        self.market_dict: Dict[MarketSegment, int] = {ms: idx for idx, ms in enumerate(market_segments)}
        self.target_dict: Dict[MarketSegment, int] = {ts: idx for idx, ts in enumerate(target_segments)}

        # self.game_features_list: List[List[float]] = []
        # self.features_list: List[List[List[float]]] = []
    
    def on_new_game(self) -> None:
        pass

    def get_ad_bids(self) -> Set[BidBundle]:
        
        self.model.eval()

        # create features        
        features = self.scale
        features_tensor = torch.tensor([features]).float()
        print(features_tensor.shape)

        with torch.no_grad():
            # Get predictions from the model
            labels = self.model(features_tensor)

        campaigns = set()

        for index in range(26):
            reach = budget = int(10000 * labels[index])
            predicted_campaign = Campaign(reach, target_segments[index], 0, 1)
            predicted_campaign.budget = budget
            predicted_campaign.uid = -(index + 1) # fake uid
            campaigns.add(predicted_campaign)
        
        # get active open campaigns
        active_campaigns: Set[Campaign] = set()
        for campaign in list(self.open_campaigns):
            if campaign.end_day < self.current_day: continue
            if campaign.start_day > self.current_day: continue
            active_campaigns.add(campaign)

        # resort to Walrasian equalibrium
        results = solution.walrasian_equilibrium(campaigns.union(active_campaigns), users)

        # make active open campaigns count list

        # my activate campaigns    
        my_active_campaigns: Set[Campaign] = set()
        for campaign in list(self.get_active_campaigns().union(self.my_campaigns)):
            if campaign.end_day < self.current_day: continue
            if campaign.start_day > self.current_day: continue
            my_active_campaigns.add(campaign)

        # a set: my campaign_uid
        my_campaigns_set: Set[int] = set()

        for my_campaign in my_active_campaigns:
            my_campaigns_set.add(my_campaign.uid)

        # a dictionary: campaign_uid -> bid_entries
        my_campaigns_bid_entries: Dict[int, Set] = {}

        # a dictionary: campaign_uid -> limit
        limit_budget: Dict[int, float] = {}
        
        for result in results:
            campaign_uid = result[0]
            if campaign_uid in my_campaigns_dict:
                auction_item=result[1]
                bid_per_item=result[2]
                bid_limit=result[3]
                bid = Bid(
                    bidder=self,
                    auction_item=auction_item,
                    bid_per_item=bid_per_item,
                    bid_limit=bid_limit
                )
                my_campaigns_bid_entries[campaign_uid].add(bid)
                
                try:
                    limit_budget[campaign_uid] += bid_limit
                except KeyError:
                    limit_budget[campaign_uid] = bid_limit
                
        
        # Bidding
        bundles = set()

        for my_campaign in my_active_campaigns:
            limit = limit_budget[my_campaign]
            bundle = BidBundle(
                campaign_id=my_campaign.uid,
                limit=limit,
                bid_entries=my_campaigns_dict[my_campaign.uid]
            )
            bundles.add(bundle)
        
        return bundles

    def get_campaign_bids(self, campaigns_for_auction:  Set[Campaign]) -> Dict[Campaign, float]:
        
        # store campaigns
        my_campaigns = self.get_active_campaigns().union(self.my_campaigns)
        self.open_campaigns = self.open_campaigns.union(campaigns_for_auction)
        self.open_campaigns = self.open_campaigns.union(my_campaigns)

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
    test_agents = [LSTMNDaysNCampaignsAgent()] + [Tier1NDaysNCampaignsAgent(name=f"Agent {i + 1}") for i in range(9)]

    # Don't change this. Adapt initialization to your environment
    simulator = AdXGameSimulator()
    simulator.run_simulation(agents=test_agents, num_simulations=10) # originally 500 