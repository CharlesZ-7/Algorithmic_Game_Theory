from adx.agents import NDaysNCampaignsAgent
from adx.tier1_ndays_ncampaign_agent import Tier1NDaysNCampaignsAgent
from adx.adx_game_simulator import AdXGameSimulator
from adx.structures import Bid, Campaign, BidBundle 
from typing import Set, Dict

class MyNDaysNCampaignsAgent(NDaysNCampaignsAgent):

    def __init__(self):
        # TODO: fill this in (if necessary)
        super().__init__()
        self.name = "My Agent"  # TODO: enter a name.

    def on_new_game(self) -> None:
        # TODO: fill this in (if necessary)
        pass

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

        # iterate over campaigns
        campaigns = self.get_active_campaigns()
        for campaign in campaigns:

            # set spending limit 
            limit = campaign.budget

            # create bids
            bid_entries = set()
            n = len(campaign.target_segment.all_segments()) # TODO: does this give overlapping segments???
            for segment in campaign.target_segment.all_segments():
                
                # auction items (market segment) # this isn't a set even though the api says it's supposed to be...
                # auction_item = set()
                # auction_item.add(segment)
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
            bids[campaign] = 1.


        return bids

if __name__ == "__main__":
    # Here's an opportunity to test offline against some TA agents. Just run this file to do so.
    test_agents = [MyNDaysNCampaignsAgent()] + [Tier1NDaysNCampaignsAgent(name=f"Agent {i + 1}") for i in range(9)]

    # Don't change this. Adapt initialization to your environment
    simulator = AdXGameSimulator()
    simulator.run_simulation(agents=test_agents, num_simulations=500)