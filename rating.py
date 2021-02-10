from  trueskill  import  Rating, quality_1vs1, rate_1vs1
import trueskill
import sys
import sqlite3
import pandas as pd
import numpy as np
import itertools
import math
import matplotlib.pyplot as plt
import mpmath

pd.options.display.max_columns = None
pd.options.mode.chained_assignment = None
pd.set_option("display.max_rows", None)

dbname = '../DB/race_joinracer_info.db'
conn = sqlite3.connect(dbname)
cursor = conn.cursor()

racer_df = pd.read_sql("SELECT * FROM racer ORDER BY ID ASC",conn,index_col="index")
racer_df = racer_df.drop_duplicates("ID")

games_df = pd.read_sql("SELECT Position,Lane,Register,raceID FROM race_result",conn)
#games_df["raceID"] = games_df["raceID"].map(lambda x: x[:-2])
games_df["raceID"] = games_df["raceID"].astype("int64")

conn.commit()
cursor.close()
conn.close()


def win_probability(team1, team2, env_=None):
    env_ = env_ if env_ else trueskill.global_env()
    delta_mu = sum(r.mu for r in team1) - sum(r.mu for r in team2)
    sum_sigma = sum(r.sigma ** 2 for r in itertools.chain(team1, team2))
    size = len(team1) + len(team2)
    denom = math.sqrt(size * (env_.beta * env_.beta) + sum_sigma)
    return env_.cdf(delta_mu / denom)

def convert_int(x):
    try:
        return int(x)
    except:
        return 6


env = trueskill.TrueSkill(beta=6.25,backend="mpmath")

ratings = {}
for i in racer_df["ID"]:
    if (i in ratings) == False:
        ratings[i] = env.create_rating()

indexs = []
mu = []
sigma = []
exp = []
win_prob =[]
quality = []
raceID  =games_df["raceID"].unique()
for group in raceID:
    df = games_df[games_df["raceID"] == group]
    IDs = df["ID"].copy()
    finishorder = df["Position"].copy().values

    rating_groups = [(ratings[i],) for i in IDs]
    rating_weight = [(0.1,) if i.startswith(('F','K','L','S')) else (1,) for i in finishorder]
    finishorder = [convert_int(i) for i in finishorder]
    for index,i in zip(IDs.index,range(len(rating_groups))):
        indexs += [index]
        mu +=[rating_groups[i][0].mu]
        sigma +=[rating_groups[i][0].sigma]
        exp += [rating_groups[i][0].exposure]
        win_prob += [win_probability(rating_groups[i],j,env) for j in rating_groups]
        quality += [env.quality(rating_groups)]

    rate_group = env.rate(rating_groups,ranks=finishorder,weights=rating_weight)
    for ID,rate_obj in zip(IDs,rate_group):
        ratings[ID] = rate_obj[0]

    

win_prob = np.reshape(np.array(win_prob),[-1,6])


rating_df = pd.DataFrame(data={"mu":mu,"sigma":sigma,"exp":exp,"quality":quality},index=indexs)
rating_df = pd.concat([rating_df,pd.DataFrame(win_prob,index=indexs).astype(float)],axis=1)
rating_df = rating_df.sort_index()


rating_df.to_csv("fixrate.csv",mode="w",encoding="utf-8-sig")




mu_ = []
sigma_ = []
exp_ = []
for value in ratings.values():
    mu_ +=[value.mu]
    sigma_ +=[value.sigma]
    exp_ += [value.exposure]
racer_df["mu"] = mu_
racer_df["sigma"] = sigma_
racer_df["exp"] = exp_
games_df.to_csv("fixrate.csv",mode="w",encoding="utf-8-sig")


    
 

