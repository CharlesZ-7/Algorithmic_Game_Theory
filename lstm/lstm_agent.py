from typing import Set, Dict, List

import sys, os
sys.path.append("..")
import random

from adx.agents import NDaysNCampaignsAgent
from adx.tier1_ndays_ncampaign_agent import Tier1NDaysNCampaignsAgent
from adx.adx_game_simulator import AdXGameSimulator
from adx.structures import Bid, Campaign, BidBundle, MarketSegment

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import LBFGS, NAdam
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.datasets as dsets

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
    MarketSegment(("Female", "Old",))
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

    def __init__(self, name = "The Agent"):
        super().__init__()
        self.name = name

        self.all_campaigns: Set[Campaign] = set() # hold all campaigns


        # import the model here

        class CampaignPredictor(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, output_size):
                super(CampaignPredictor, self).__init__()
                self.num_layers = num_layers
                self.hidden_size = hidden_size
        
                # LSTM with specified number of layers and neurons
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                
                # Fully connected output layer
                self.fc = nn.Linear(hidden_size, output_size)
        
                # Xavier normal initialization
                self.init_weights()
        
            def init_weights(self):
                for name, param in self.lstm.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_normal_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.xavier_normal_(param.data)
                nn.init.xavier_normal_(self.fc.weight)
        
            def forward(self, x):
                # Initialize hidden and cell states
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
                # Forward propagate the LSTM
                out, _ = self.lstm(x, (h0, c0))
                
                # Pass the output of the last time step to the classifier
                out = self.fc(out[:, -1, :])
                return torch.sigmoid(out)  # Softmax is applied later in the training loop

        class CampaignDataset(Dataset):
            def __init__(self, sequences):
                self.sequences = sequences
        
            def __len__(self):
                return len(self.sequences)
        
            def __getitem__(self, index):
                sequence = self.sequences[index]
                # Assuming each sequence is a tuple (features, target)
                features = torch.tensor(sequence[0], dtype=torch.float32)
                target = torch.tensor(sequence[1], dtype=torch.float32)
                return features, target
    
    def on_new_game(self) -> None:
        pass

    def get_ad_bids(self) -> Set[BidBundle]:

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
            
        # lstm bidding
            # TODO

            # self.run model...

        input_dim = 34
        output_dim = 26
        hidden_dim = 100
        layer_dim = 3
        NAdam_iter = 5000
        LBFGS_iter = 3000
        batch_size = 1000
        
        criterion = nn.BCEWithLogitsLoss()
        NAdam_optimizer = NAdam(model.parameters(), lr=0.001)
        LBFGS_optimizer = LBFGS(model.parameters())

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = CampaignPredictor(input_size, hidden_size, num_layers, output_size)
        train(model, NAdam_optimizer, NAdam_iter, train_loader, epochs, device=device)
        train(model, LBFGS_optimizer, LBFGS_iter, train_loader, epochs, device=device)
        evaluate(model, test_loader, device=device)

        
        
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
    test_agents = [LSTMNDaysNCampaignsAgent()] + [Tier1NDaysNCampaignsAgent(name=f"Agent {i + 1}") for i in range(9)]

    # Don't change this. Adapt initialization to your environment
    simulator = AdXGameSimulator()
    simulator.run_simulation(agents=test_agents, num_simulations=500) # originally 500 