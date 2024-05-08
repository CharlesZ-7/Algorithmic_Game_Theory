import matplotlib.pyplot as plt
import numpy as np
import json

# # graphing training curves

# data_min = []
# data_max = []
# data_mean = []
# with open("training_log.txt", "r") as f:
#     for x in f:
#         data = json.loads(x)
#         data_min.append(data["episode_reward_min"])
#         data_max.append(data["episode_reward_max"])
#         data_mean.append(data["episode_reward_mean"])

# n = len(data_mean)
# x = np.arange(n) * 4000
# y = data_min
# plt.plot(x, y)

# y = data_max
# plt.plot(x, y)

# y = data_mean
# plt.plot(x, y)

# plt.legend(["Min", "Max", "Mean"])

# plt.xlabel("Number of Games Played")
# plt.ylabel("Normalized Profits")
# plt.title("Training Curves for RL Agent")

# # plt.show() 

# plt.savefig("training_curves.png")


# graph agent information

# path = "data_no_rl.json"
path = "data_tier_1.json"
# path = "data.json"
with open(path, "r") as f:
    data = json.load(f)

for key in data:

    # if key != "WF Agent": continue

    # y = np.cumsum(data[key]["profits"])
    # y = data[key]["quality_scores"]
    y = np.cumsum(data[key]["campaign_auction_winnings"])
    x = np.arange(len(y))
    line, = plt.plot(x, y)

    if key == "WF Agent":
        line.set_label("WF Agent")

# labelling

# plt.title("Games Played against Cumulative Profits")
# plt.ylabel("Cumulative Profits")

# plt.title("Games Played against Quality Score")
# plt.ylabel("Quality Score")

plt.title("Games Played against Cumulative Sum of Auctions Won")
plt.ylabel("Cumulative Sum of Auctions Won")



plt.xlabel("Number of Games Played")
plt.legend()
plt.tight_layout()


# plt.savefig("profits_tier_1.png")
# plt.savefig("quality_scores_tier_1.png")
plt.savefig("campaigns_tier_1.png")


plt.show()
