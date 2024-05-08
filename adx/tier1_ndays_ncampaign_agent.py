import random
from typing import Set, Dict, List
from adx.structures import Bid, Campaign, BidBundle, MarketSegment
from adx.agents import NDaysNCampaignsAgent

class Tier1NDaysNCampaignsAgent(NDaysNCampaignsAgent):
    def __init__(self, name):
        super().__init__()
        self.name = name

        self.auctions = 0
        self.auctions_won = 0
        self.campaign_auction_history: List[Campaign] = []

        # logging information for graphing...
        self.profit_history: List[float] = []
        self.campaign_winning_history: List[int] = []
        self.quality_score_history = []

    def log_stats(self, campaigns_for_auction):

        # print(self.current_day)

        if self.current_day >= 9:
            self.profit_history.append(self.profit)
            self.campaign_winning_history.append(self.auctions_won)
            self.quality_score_history.append(self.quality_score)

    def count_won_auctions(self, campaigns_for_auction:  Set[Campaign]):
        
        # count number of won auctions
        self.auctions += len(self.campaign_auction_history)
        my_campaigns = self.get_active_campaigns().union(self.my_campaigns)
        for campaign in self.campaign_auction_history:
            if campaign in my_campaigns:
                self.auctions_won += 1

        # store campaigns for next iteration
        self.campaign_auction_history = campaigns_for_auction

    def get_ad_bids(self) -> Set[BidBundle]:
        bundles = set()
        for campaign in self.get_active_campaigns():
            bids = set()
            bid_per_item = min(1, max(0.1, (campaign.budget - self.get_cumulative_cost(campaign)) /
                               (campaign.reach - self.get_cumulative_reach(campaign) + 0.0001)))
            total_limit = max(1.0, campaign.budget - self.get_cumulative_cost(campaign))
            auction_item = campaign.target_segment
            bid = Bid(self, auction_item, bid_per_item, total_limit)
            bids.add(bid)
            bundle = BidBundle(campaign_id=campaign.uid, limit=total_limit, bid_entries=bids)
            bundles.add(bundle)
        return bundles

    def get_campaign_bids(self, campaigns_for_auction: Set[Campaign]) -> Dict[Campaign, float]:
        self.log_stats(campaigns_for_auction)
        self.count_won_auctions(campaigns_for_auction)
        
        bids = {}
        for campaign in campaigns_for_auction:
            bid_value = campaign.reach * (random.random() * 0.9 + 0.1)
            bids[campaign] = bid_value
        return bids

    def on_new_game(self):
        pass


