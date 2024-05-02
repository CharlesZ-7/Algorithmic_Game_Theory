from adx.agents import NDaysNCampaignsAgent
from adx.tier1_ndays_ncampaign_agent import Tier1NDaysNCampaignsAgent
from adx.adx_game_simulator import AdXGameSimulator
from adx.structures import Bid, Campaign, BidBundle, MarketSegment
from typing import Set, Dict, List

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

class MyNDaysNCampaignsAgent(BetterNDaysNCampaignsAgent):

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

    # def __init__(self):
    #     # TODO: fill this in (if necessary)
    #     super().__init__()
    #     self.name = "My Agent"  # TODO: enter a name.

    # def on_new_game(self) -> None:
    #     # TODO: fill this in (if necessary)
    #     pass

    # def get_ad_bids(self) -> Set[BidBundle]:
    #     # TODO: fill this in
        
    #     bundles = set()

    #     # notes:

    #     # if there are two campaigns with two auctions...
    #         # which does it prioritize...
    #         # take into consideration the profit...
    #         # take into account the effects on the quality score...

    #         # prioritize the campaign with lower effective reach...

    #     # during the last days of the campaign, we don't care about the quality score...

    #     # ... 



    #     # --- basic syntax to create and send bids ---

    #     # all campaigns, campaigns from auction and random campaigns
    #     campaigns = self.get_active_campaigns().union(self.my_campaigns)

    #     # iterate over campaigns
    #     for campaign in campaigns:

    #         # create bids
    #         bid_entries = set()

    #         # find all subsets
    #         subsets = set()
    #         for segment in MarketSegment.all_segments():
    #             if campaign.target_segment.issuperset(segment):
    #                 subsets.add(segment)

    #         # iterate over subsets
    #         n = len(subsets)
    #         for segment in subsets:
                
    #             # auction items (market segment) # this is a market segment...
    #             auction_item = segment
                
    #             # bid per item
    #             bid_per_item = campaign.budget / (n * campaign.reach)

    #             # create bid object
    #             bid = Bid(
    #                 bidder=self,
    #                 auction_item=auction_item,
    #                 bid_per_item=bid_per_item,
    #                 bid_limit=campaign.budget / n
    #             )
    #             bid_entries.add(bid)

    #         # set spending limit 
    #         limit = campaign.budget

    #         # bundle bid and return
    #         bundle = BidBundle(
    #             campaign_id=campaign.uid,
    #             limit=limit,
    #             bid_entries=bid_entries
    #         )
    #         bundles.add(bundle)

    #     return bundles

    # def get_campaign_bids(self, campaigns_for_auction:  Set[Campaign]) -> Dict[Campaign, float]:
    #     # TODO: fill this in 
        
            
    #     bids = {}

    #     # notes:

    #     # if agent already owns a campaign, should it go for a monopoly or diversify?
    #         # try to win campaigns for similar market sections to block competition
    #         # don't get too many campaigns otherwise quality scores will go down

    #     # assign campaigns some kind of score for how much we want to bid on it...

    #     # try to prioritize shorter campaigns... retain high effective reach...
    #         # just be sure that it is still possible to fufill...

    #     # if we know what other campaigns are going on, go for the less competitive ones...

    #     # how low to bid on an auction as this determines the budget for that auction...
    #         # some kind of ai here?? we need better ai's 
    #         # some kind of history...
    #         # lenght of contracts... fill these as fast as we can...

    #     # ... 



    #     # --- basic syntax to create and send bids ---

    #     # iterate over campaigns
    #     for campaign in campaigns_for_auction:
    #         bids[campaign] = 1000.

    #     return bids

if __name__ == "__main__":
    # Here's an opportunity to test offline against some TA agents. Just run this file to do so.
    test_agents = [MyNDaysNCampaignsAgent()] + [Tier1NDaysNCampaignsAgent(name=f"Agent {i + 1}") for i in range(9)]

    # Don't change this. Adapt initialization to your environment
    simulator = AdXGameSimulator()
    simulator.run_simulation(agents=test_agents, num_simulations=1) # originally 500 