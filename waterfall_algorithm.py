from typing import Set
from adx.structures import Campaign, MarketSegment
import numpy as np

# This uses the waterfall algorithm to calculate optimal prices for a
# campaign in a second price auction.


# so far this only creates the allocation and price tables...


def market_segment_to_key(segment: MarketSegment):
    output = ""
    for item in segment:
        output += item[0]
    return output

def run(campaigns: Set[Campaign]):

    # user_frequencies = {
    #     MarketSegment(("Male", "Young", "LowIncome")): 1836,
    #     MarketSegment(("Male", "Young", "HighIncome")): 517,
    #     MarketSegment(("Male", "Old", "LowIncome")): 1795,
    #     MarketSegment(("Male", "Old", "HighIncome")): 808,
    #     MarketSegment(("Female", "Young", "LowIncome")): 1980,
    #     MarketSegment(("Female", "Young", "HighIncome")): 256,
    #     MarketSegment(("Female", "Old", "LowIncome")): 2401,
    #     MarketSegment(("Female", "Old", "HighIncome")): 407
    # }

    user_frequencies = {
        MarketSegment(("Male",)): 8,
        MarketSegment(("Female",)): 7
    }




    # create a tuple list of critical bids
    critical_bids = []
    for campaign in campaigns:
        critical_bid = campaign.budget / campaign.reach
        critical_bids.append((critical_bid, campaign))

    # sort these campaigns from highest to lowest
    critical_bids.sort(key=lambda input: input[0], reverse=True)

    
    print(critical_bids)





    # # get a list of all the critical bids
    # count = 0
    # critical_bids = []
    # campaign_indices = []
    # campaigns_list = []
    # for campaign in campaigns:
    #     campaigns_list.append(campaign)
    #     campaign_indices.append(count)
    #     count += 1
    #     critical_bids.append(campaign.budget / campaign.reach)

    #     marketsegments = []
    #     for target_user in user_frequencies:
    #         if campaign.target_segment.issuperset(target_user):
    #             marketsegments.append(target_user)

    # print(critical_bids)

    # # allocation and price tables
    # allocation_price_dict = {}

    # # iterate over highest
    # critical_bids = np.array(critical_bids)
    # campaign_indices = np.array(campaign_indices)

    # # sort the campaigns by critical bids




    # while (len(campaign_indices > 0)):


    #     arg_max = np.argmax(critical_bids)
    #     critical_bid = critical_bids[arg_max]
    #     campaign_index = campaign_indices[arg_max]
    #     market_segment : MarketSegment = campaigns_list[campaign_index].target_segment

    #     critical_bids = np.delete(critical_bids, arg_max)
    #     campaign_indices = np.delete(campaign_indices, arg_max)


    #     # might as well just sort them all and then iterate through using the stuff...
    #     # use tuples and stuff instead...


    #     # find the second highest in a competing market segment
    #     campaign_search_critical_bids = critical_bids
    #     campaign_search_campaign_indices = campaign_indices
    #     second_price = 0
    #     while (len(campaign_search_campaign_indices) > 0):
    #         campaign_search_arg_max = np.argmax(campaign_search_critical_bids)
    #         campaign_search_campaign_index = campaign_indices[campaign_search_arg_max]
    #         campain_search_market_segment : MarketSegment = campaigns_list[campaign_search_campaign_index].target_segment
    #         campaign_search_critical_bid = campaign_search_critical_bids[campaign_search_arg_max]
    #         campaign_search_critical_bids = np.delete(campaign_search_critical_bids, campaign_search_arg_max)
    #         campaign_search_campaign_indices = np.delete(campaign_search_campaign_indices, campaign_search_arg_max)


    #         # be more precise here...
    #         # there are cases when stuff is split and prices are off and whatever...

    #         # there are just more edge cases and cost splits possible...


    #         # check for overlapping segments
    #         if not market_segment.issubset(campain_search_market_segment) and not market_segment.issuperset(campain_search_market_segment): continue

    #         # found competing segment
    #         second_price = campaign_search_critical_bid

    #     # find all subsegments
    #     sub_market_segments = []
    #     for segment in user_frequencies:
    #         if market_segment.issuperset(segment):
    #             sub_market_segments.append(segment)

    #     # distribute users
    #     n = np.floor(campaigns_list[campaign_index].reach / len(sub_market_segments))
    #     allocations = []
    #     for segment in sub_market_segments:
    #         min = np.min((user_frequencies[segment], n))
    #         user_frequencies[segment] -= min
    #         allocations.append((segment, min))
    #     allocation_price_dict[campaigns_list[campaign_index].uid] = allocations, second_price

    # print(allocation_price_dict)


if __name__ == "__main__":

    # example from paper

    # campaign_0 = Campaign(10, MarketSegment(("Male", "Female")), 0, 1) # reach, target, start, end
    # campaign_0.budget = 100
    # campaign_1 = Campaign(5, MarketSegment(("Male", "Female")), 0, 1) # reach, target, start, end
    # campaign_1.budget = 25

    # campaigns = set()
    # campaigns.add(campaign_0)
    # campaigns.add(campaign_1)

    # run(campaigns)



    # personal example

    campaign_0 = Campaign(10, MarketSegment(("Male", "Young", "LowIncome")), 0, 2) # reach, target, start, end
    campaign_0.budget = 10
    campaign_1 = Campaign(11, MarketSegment(("Female", "LowIncome")), 0, 1) # reach, target, start, end
    campaign_1.budget = 15
    campaign_2 = Campaign(12, MarketSegment(("Male", "Young")), 0, 3) # reach, target, start, end
    campaign_2.budget = 5

    campaigns = set()
    campaigns.add(campaign_0)
    campaigns.add(campaign_1)
    campaigns.add(campaign_2)

    run(campaigns)

    # print(market_segment_to_key(MarketSegment(("Male", "Young", "LowIncome"))))



    # TODO:
        # debug this a little bit with the added complexity...
            # maybe even use a random number generator to split if the prices on the things are equal...
            # incorporate the days into the thingy...

        # figure out how to train the rl better...
        # get agents that are decently smart to train against...

        # smartest agent without using rl...

        # figure out different possible strategies for this...
            # 2 max random agents...
            # 2 critical bidding agents
            # rest are very smart agents...

        # get the rl to predict campaigns out there...
            # if this proves to be too difficult of a problem, just run the WF or WE on what we do know and then let rl to shade shit higher or lower...

        
        # more history information neeeded... figure out if we won the auctions or not...
        # if we did lose that stuff then inputs given as shtuff...

        # tuning hyperparameters... higher learning rate because we don't have 1million training iterations...




        # what's the difference between active campaigns and my campaigns...
        # why tf does this not include the first campaign... sort these out by the day to filter stuff out...