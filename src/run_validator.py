import os

def is_valid(dir):
    return True
    # if (dir.endswith ("-auto")):
    #     print(f"rejected {dir}")
    #     return False
    # try:
    #     with open(os.path.join(dir, "metadata.txt")) as f:
    #         lines = f.readlines()
    # except:
    #     return False
    
    # eliminations_overall = []
    # for line in lines:
    #     index, time, eliminations = line.split(';')
    #     index = int(index.strip())
    #     time = float(time.strip())
    #     eliminations = eliminations.strip("[] \n").split(",")
    #     eliminations = [int(x) for x in eliminations]
    #     eliminations_overall.append(eliminations)

    # actions = [act for elm in eliminations_overall for act in range(0, 5) if act not in elm]
    
    # if (len(actions) == 0):
    #     return False
    # # print(f"{dir}: ", len(actions))
    # nothing_count = actions.count(0)
    # action_rate = (len(actions) - nothing_count) / len(actions)
    # action_count = len(actions) - nothing_count

    # return action_rate > 0.25 and action_count > 10 and len(actions) > 50