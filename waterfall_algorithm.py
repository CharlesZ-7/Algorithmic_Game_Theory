from typing import Set, Dict, List, Tuple
from adx.structures import Campaign, MarketSegment
import numpy as np

# Given campaigns, calculates alllocation and price tables

# TODO:


    # how to get this to beat the normal agents...

    # get a specific agent to beat out the critical bidding agents...




    # figure out how to train the rl better...
    # get agents that are decently smart to train against...

    # smartest agent without using rl...

    # figure out different possible strategies for this...
        # 2 max random agents...
        # 2 critical bidding agents
        # rest are very smart agents...





    # get the rl to predict campaigns out there...
        # if this proves to be too difficult of a problem, just run the WF or WE on what we do know and then let rl to shade shit higher or lower...

        # simplify this problem by only caring about stuff...
        # wait why cant we just use our knowledge to guess this stuff...

        # no reason to have rl deal with this...



    
    # more history information neeeded... figure out if we won the auctions or not...
    # if we did lose that stuff then inputs given as shtuff...

    # tuning hyperparameters... higher learning rate because we don't have 1million training iterations...




    # what's the difference between active campaigns and my campaigns...
    # why tf does this not include the first campaign... sort these out by the day to filter stuff out...







def run(day: int, campaigns: Set[Campaign]) -> Dict[Campaign, List[Tuple[MarketSegment, float, int]]]:

    # frequencies for the day
    user_frequencies = {
        MarketSegment(("Male", "Young", "LowIncome")): 1836,
        MarketSegment(("Male", "Young", "HighIncome")): 517,
        MarketSegment(("Male", "Old", "LowIncome")): 1795,
        MarketSegment(("Male", "Old", "HighIncome")): 808,
        MarketSegment(("Female", "Young", "LowIncome")): 1980,
        MarketSegment(("Female", "Young", "HighIncome")): 256,
        MarketSegment(("Female", "Old", "LowIncome")): 2401,
        MarketSegment(("Female", "Old", "HighIncome")): 407
    }

    # debugging example
    user_frequencies = {
        MarketSegment(("Male", "Young", "LowIncome")): 10,
        MarketSegment(("Male", "Young", "HighIncome")): 517,
        MarketSegment(("Male", "Old", "LowIncome")): 1795,
        MarketSegment(("Male", "Old", "HighIncome")): 808,
        MarketSegment(("Female", "Young", "LowIncome")): 1980,
        MarketSegment(("Female", "Young", "HighIncome")): 256,
        MarketSegment(("Female", "Old", "LowIncome")): 2401,
        MarketSegment(("Female", "Old", "HighIncome")): 407
    }
    # user_frequencies = {
    #     MarketSegment(("Male",)): 8,
    #     MarketSegment(("Female",)): 7
    # }

    # remove campaigns if their end date has passed
    reduced_campaigns: Set[Campaign] = set()
    for campaign in campaigns:
        if campaign.end_day < day: continue
        reduced_campaigns.add(campaign)
    campaigns = reduced_campaigns
    
    # create a tuple list of critical bids
    critical_bids = []
    for campaign in campaigns:
        critical_bid = campaign.budget / campaign.reach
        critical_bids.append((critical_bid, campaign))

    # sort these campaigns from highest to lowest
    critical_bids.sort(key=lambda input: input[0], reverse=True)

    # allocations 
    allocation_table = {}
    for i in range(len(critical_bids)):
        critical_bid, campaign = critical_bids[i]

        # find all sub-market segments
        sub_market_segments = []
        for segment in user_frequencies:
            if segment.issuperset(campaign.target_segment):
                sub_market_segments.append((0, segment))

        # find second highest bid for each sub-market segment
        for j in range(len(sub_market_segments)):
            _, segment = sub_market_segments[j]
            for k in range(i+1, len(critical_bids)):
                second_critical_bid, second_campaign = critical_bids[k]
                if segment.issubset(second_campaign.target_segment) or segment.issuperset(second_campaign.target_segment):
                    sub_market_segments[j] = second_critical_bid, segment
                    break

        # sort sub market segments 
        sub_market_segments.sort(key=lambda input: input[0])

        # distribute uesrs with the lowest price first
        reach = campaign.reach
        total_reach = 0
        index = 0
        campaign_allocations = []
        while total_reach < reach:
            if index >= len(sub_market_segments): break
            price, segment = sub_market_segments[index]
            allocation = min(user_frequencies[segment], reach - total_reach)
            user_frequencies[segment] -= allocation
            total_reach += allocation
            campaign_allocations.append((segment, price, allocation))
            index += 1
        allocation_table[campaign] = campaign_allocations
    
    # return allocation table
    return allocation_table



if __name__ == "__main__":

    # example from paper

    # campaign_0 = Campaign(10, MarketSegment(("Male", "Female")), 0, 1) # reach, target, start, end
    # campaign_0.budget = 100
    # campaign_1 = Campaign(5, MarketSegment(("Male", "Female")), 0, 1) # reach, target, start, end
    # campaign_1.budget = 25

    # campaigns = set()
    # campaigns.add(campaign_0)
    # campaigns.add(campaign_1)

    # run(0, campaigns)


    # personal example

    campaign_0 = Campaign(10, MarketSegment(("Male",)), 0, 2) # reach, target, start, end
    campaign_0.budget = 10
    campaign_1 = Campaign(11, MarketSegment(("Male", "LowIncome")), 0, 1) # reach, target, start, end
    campaign_1.budget = 15
    campaign_2 = Campaign(12, MarketSegment(("Male", "Young", "LowIncome")), 0, 3) # reach, target, start, end
    campaign_2.budget = 5

    campaigns = set()
    campaigns.add(campaign_0)
    campaigns.add(campaign_1)
    campaigns.add(campaign_2)

    print(run(0, campaigns))