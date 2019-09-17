
import json
import numpy as np
import math
import bisect
import random

# global variables
MAX_MINION_PER_PARTY = 5


# load and preprocess minion and skill data

def generate_stats_coeff(minion): #randomize initial substats and attributes by 10%
    stats_coeff = np.array(minion['stats_coeff'])
    rand_mat = np.array([random.uniform(0.9,1.1) for _ in range(len(stats_coeff[1]))])
    stats_coeff[1] = np.multiply(stats_coeff[1], rand_mat)
    stats_coeff[1][0:2] = np.floor(stats_coeff[1][0:2]) + 1
    stats_coeff[1][4:7] = np.floor(stats_coeff[1][4:7]) + 1
    minion['stats_coeff'] = stats_coeff
    return minion


def generate_luck(minion, p = [0.2, 0.65, 0.15], magnitude = 0.3):
    # p: luck distribution, [bad, neutural, good]
    n_stats = len(minion['base_stats'])
    n_attributes = len(minion['base_attributes'])
    #n_skills = len(minion['smoves'])
    n_abilities = len(minion['abilities'])
    
    stats_luck = np.array([np.random.choice(1+np.arange(-1,2)*magnitude, p=p) for _ in range(n_stats)])
    attributes_luck = np.array([np.random.choice(1+np.arange(-1,2)*magnitude, p=p) for _ in range(n_attributes)])
    #skills_luck = np.array([np.random.choice(1-np.arange(-1,2)*magnitude, p=p) for _ in range(n_skills)])
    smove_luck = np.random.choice(1-np.arange(-1,2)*magnitude, p=p)
    abilities_luck = np.array([np.random.choice(1-np.arange(-1,2)*magnitude, p=p) for _ in range(n_abilities)])

    luck = np.array([stats_luck, attributes_luck, smove_luck, abilities_luck])
    #luck = np.array([stats_luck, attributes_luck, skills_luck, abilities_luck])
    
    minion['luck'] = luck
    return minion


def compute_current_status(minion): # f(base_stats, base_attributes, luck, wellness_factor)
    
    def clip_by_capacity(attr, attr_capacity):
        for i in range(len(attr)):
            attr[i] = min(attr[i], attr_capacity[i])
        return attr
    
    base_stats = np.array(minion['base_stats'])
    base_attributes = np.array(minion['base_attributes'])
    luck = np.array(minion['luck'])
    wellness_factor = minion['wellness_factor']
    stats_coeff = np.array(minion['stats_coeff'])
    
    # base_capacity = f(base_stats)
    base_capacity = np.floor(np.multiply(base_stats, stats_coeff[0]) + stats_coeff[1]) + 1
    base_substats = base_capacity[0:4]
    base_attributes_capacity = base_capacity[4:7]
    base_attributes = clip_by_capacity(base_attributes, base_attributes_capacity)
    
    # cur_stats = f(base_stats, luck, wellness_factor)
    cur_stats = np.floor((np.multiply(base_stats, luck[0]) * wellness_factor)) + 1
    
    # cur_capacity = f(cur_stats)
    cur_capacity = np.floor(np.multiply(cur_stats, stats_coeff[0]) + stats_coeff[1]) + 1
    cur_substats = cur_capacity[0:4]
    cur_attributes_capacity = cur_capacity[4:7]
    
    # cur_attributes = f(base_attributes, luck)
    cur_attributes = clip_by_capacity(np.floor(np.multiply(base_attributes, luck[1])) + 1, cur_attributes_capacity)
    
    # tbd: cp = f(cur_substats, cur_attributes, skill's DPS, ...)
    # current: cp = strength * sqrt(endurance) * sqrt(constitution)
    cp = cur_stats[0] * cur_stats[1]**(1/2) * cur_stats[3]**(1/2)
    
    # update
    minion['base_substats'] = base_substats
    minion['base_attributes_capacity'] = base_attributes_capacity
    minion['base_attributes'] = base_attributes
    minion['cur_stats'] = cur_stats
    minion['cur_substats'] = cur_substats
    minion['cur_attributes_capacity'] = cur_attributes_capacity
    minion['cur_attributes'] = cur_attributes
    minion['cp'] = cp

    return minion

def load_minion_data(fn, update_luck = True, update_stats_coeff = True):
    #read minion data from json
    #(optional) generate luck, randomize stats coefficient
    #compute current status, index on mid, output dict
    
    minion_data = {}
    
    with open(fn,'r') as f:
        d = json.load(f)
            
    for m in d:
        if update_luck or len(m['luck']) == 0:
            m = generate_luck(m)
        if update_stats_coeff:
            m = generate_stats_coeff(m)
        m = compute_current_status(m)
        minion_data[m['id']] = m
    return minion_data


def load_data(fn):
    data = {}
    
    with open(fn,'r') as f:
        d = json.load(f)
    
    for l in d:
        data[l['id']] = l
    
    return data



# battle classes

class Skill:

    def __init__(self, **kwargs):
    # kwargs: id, name, power, mpdelta, cast, cd, mtype
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.free_since = 0


class BattleMinion:
    
    def __init__(self, minion_dict, skill_dict, mid):
        this_minion = minion_dict[mid]
        self.id = this_minion['id']
        self.name = this_minion['name']
        self.attack = this_minion['cur_substats'][0]
        self.defense = this_minion['cur_substats'][1]
        self.critical = this_minion['cur_substats'][2]
        self.hit = this_minion['cur_substats'][3]
        self.hp = this_minion['cur_attributes'][0]
        self.mp = this_minion['mp']
        self.mp_capacity = this_minion['mp_capacity']
        self.mp_rate1 = this_minion['mp_regen_per_second']
        self.mp_rate2 = this_minion['mp_gain_rate_damage_dealt']
        self.mp_rate3 = this_minion['mp_gain_rate_damage_taken']
        self.luck = this_minion['luck']
        
        self.fmoves = {}
        self.smoves = {}
        
        for sid in this_minion['fmoves']:
            move = Skill(**skill_dict[sid])
            move.dps = move.power/(move.cast+move.cd)
            self.fmoves[move.id] = move
        
        smove_min_mp = 0
        for sid in this_minion['smoves']:
            move = Skill(**skill_dict[sid])
            move.mpdelta = np.ceil(move.mpdelta*self.luck[2]) #smove mp cost depends on luck
            move.dps = move.power/(move.cast+move.cd)
            move.dpm = move.dps/abs(move.mpdelta)
            self.smoves[move.id] = move
            if smove_min_mp < abs(move.mpdelta):
                smove_min_mp = abs(move.mpdelta)
        self.smove_min_mp = smove_min_mp
            
    def alive(self):
        return self.hp > 0

    
#tbd: team rule
class Party(): 
    # dict of mid:mnn pairs
    def __init__(self, mnn_list=[]):
        self.dict = {}
        for mnn in mnn_list:
            self.add(mnn)

    def __iter__(self):
        return iter(self.dict)
    
    def add(self, mnn):
        # Add mnn to the party
        if len(self.dict) < MAX_MINION_PER_PARTY:
            mnn.parent_party = self
            self.dict[mnn.id] = mnn
        else:
            raise Exception("Failed to add new minion: Exceeding max party size")

    def alive(self):
        # If any of the minion in this party is still alive, returns True. 
        # Otherwise returns False
        for mid, mnn in self.dict.items():
            if mnn.hp > 0:
                return True
        return False


class Event:
    '''
    events sit on the timeline and carry instructions about what they should do.
    events do not directly manipulate minion's attributes
    
    TYPE OF EVENT
    mnn_fFree, mnn_sFree: Attacking minion is free to perform a fmove/smove. update: move.free_since
                takes (name, t, mnn_usedAtk, move, mtype)
    mnnHurt: Attacking minion makes damage and gains mp (fmove) or loses mp (smove). 
                Defending minion takes damage and gains mp.
                takes (name, t, mnn_usedAtk, mnn_hurt, damage, atkr_mpdelta, dfdr_mpdelta, move, mtype)
    mpRegen: mp regeneration for all minions per second. update: all mnn.mp += mnn.mp_rate_1
                takes (name, t)
    '''
    
    def __init__(self, name, t, mnn_usedAtk=None, mnn_hurt=None, move=None, mtype=None,
                 damage=None, atkr_mpdelta=None, dfdr_mpdelta=None, msg=None):
        self.name = name
        self.t = t
        self.mnn_usedAtk = mnn_usedAtk # attacker
        self.move = move
        self.mtype = mtype

        # for mnnHurt only:
        self.mnn_hurt = mnn_hurt # defender
        self.damage = damage
        self.atkr_mpdelta = atkr_mpdelta # this may be negative
        self.dfdr_mpdelta = dfdr_mpdelta
        
        if 'Free' in self.name:
            self.msg = '%s %s is freed' % (mnn_usedAtk.name, move.name)
        elif self.name == 'mnnHurt':
            self.msg = '%s attacks %s by %s, damage: %d' % (mnn_usedAtk.name, mnn_hurt.name, move.name, int(damage))
        
    def __lt__(self, other): return self.t < other.t
    def __le__(self, other): return self.t <= other.t
    def __gt__(self, other): return self.t > other.t
    def __ge__(self, other): return self.t >= other.t
    def __eq__(self, other): return self.t == other.t
    

class Timeline:
    # A ordered queue to manage events
    
    def __init__(self):
        self.lst = []

    def __iter__(self):
        return iter(self.lst)

    def add(self, e):
        # add the event at the proper (sorted) spot.
        bisect.insort_right(self.lst, e)

    def pop(self):
        # return the first (earliest) event and remove it from the queue
        return self.lst.pop(0)

    def print(self):
        print("==Timeline== ", end="")
        for e in self.lst:
            print(str(e.t) + ":" + e.name, end=", ")
        print()

      
class World:
    # holds all settings
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    

# battle functions

def compute_damage(atkr, dfdr, move):

    def critical_multiplier(critical_rate): # yes: 2 (critical hit creates 2x damage); no: 1
        return np.random.choice(np.arange(1,3), p=[1-critical_rate, critical_rate])
    
    def hit_multiplier(hit_rate): # yes: 1; no: 0
        return np.random.choice(np.arange(0,2), p=[1-hit_rate, hit_rate])
    
    # tbd: damage function
    damage = np.ceil(atkr.attack/dfdr.defense * critical_multiplier(atkr.critical) * move.power) * hit_multiplier(atkr.hit)

    # compute mpdelta
    if move.mtype == 's':  # for atkr, smove consumes mp
        atkr_mpdelta = move.mpdelta
    else:  # for atkr, fmove builds up mp
        atkr_mpdelta = np.ceil(damage*atkr.mp_rate2)    
    
    # for dfdr, taking damage builds up mp
    dfdr_mpdelta = np.ceil(damage*dfdr.mp_rate3)
    
    return damage, atkr_mpdelta, dfdr_mpdelta


def mnn_use_move(mnn_usedAtk, mnn_hurt, move, mtype, wd, t): 
    #compute damage, atkr_mpdelta, dfdr_mpdelta
    #add events: mnnFree, mnnHurt
    
    damage, atkr_mpdelta, dfdr_mpdelta = compute_damage(mnn_usedAtk, mnn_hurt, move)

    if move.mtype == 'f':
        wd.tline.add(Event("mnn_fFree", t + move.cd, mnn_usedAtk=mnn_usedAtk, move=move, mtype=mtype))
    else:
        wd.tline.add(Event("mnn_sFree", t + move.cd, mnn_usedAtk=mnn_usedAtk, move=move, mtype=mtype))
    wd.tline.add(Event("mnnHurt", t + move.cast, mnn_usedAtk=mnn_usedAtk, mnn_hurt=mnn_hurt, damage=damage, atkr_mpdelta=atkr_mpdelta, dfdr_mpdelta=dfdr_mpdelta, move=move, mtype=mtype))
    

#tba: mp regeneration overtime
#def regen_mp(wd, t):
#    
#    wd.tline.add(Event("mpRegen", t))


def select_dfdr(dfdr_party):
    # select lowest hp minion in opponent team to attack

    dfdr_min_hp = float('inf')
    for mid, mnn in dfdr_party.dict.items():
        if mnn.hp > 0 and mnn.hp < dfdr_min_hp:
            dfdr = mnn
    return dfdr


def player_AI_choose(wd, atkr, t, event_name):
    # This function is the strategy of the attacking minion.
    # It manipulates the timeline by inserting events,
    # will call mnn_use_move() if a best move is found
        
    def search_smoves(moveset, t, mp, smove_min_mp):
        if mp < smove_min_mp:
            return None, None
        
        available_smoves = {}
        for sid, smove in moveset.items():
            if mp + smove.mpdelta >= 0 and smove.free_since <= t:
                available_smoves[sid] = smove.dpm
        if len(available_smoves) >= 2:     # 2+ smoves available: sort by dpm (damage per mana = power/(cast + cd)/abs(mpdelta))
            sid = max(available_smoves, key=available_smoves.get)
        elif len(available_smoves) == 1:
            sid = list(available_smoves.keys())[0]
        else:  # no smove available
            sid = None
        return sid, 's'
    
    def search_fmoves(moveset, t):
        available_fmoves = {}
        for sid, fmove in moveset.items():
            if fmove.free_since <= t:
                available_fmoves[sid] = fmove.dps
        if len(available_fmoves) >= 2:    # 2+ fmoves available: sort by dps (damage per second = power/(cast + cd))
            sid = max(available_fmoves, key=available_fmoves.get)
        elif len(available_fmoves) == 1:
            sid = list(available_fmoves.keys())[0]
        else:
            sid = None
        return sid, 'f'
    
    
    sid = None
    if event_name == 'mnn_fFree':
        sid, mtype = search_fmoves(atkr.fmoves, t)
    elif event_name == 'mnn_sFree':
        sid, mtype = search_smoves(atkr.smoves, t, atkr.mp, atkr.smove_min_mp)
    elif event_name == 'mnnHurt':
        sid, mtype = search_smoves(atkr.smoves, t, atkr.mp, atkr.smove_min_mp)
        
    if sid != None:
        if mtype == 'f':
            move = atkr.fmoves[sid]
        else:
            move = atkr.smoves[sid]
        
        p1 = wd.party1
        p2 = wd.party2
        dfdr_party = p2 if atkr.parent_party==p1 else p1
        dfdr = select_dfdr(dfdr_party)
        
        mnn_use_move(atkr, dfdr, move, mtype, wd, t)


def init_atkr_event(p, tline): # free all moves

    for mid, mnn in p.dict.items():
        for sid, fmove in mnn.fmoves.items():
            tline.add(Event('mnn_fFree', 0, mnn_usedAtk=mnn, move=fmove, mtype=fmove.mtype))
        for sid, smove in mnn.smoves.items():
            tline.add(Event('mnn_sFree', 0, mnn_usedAtk=mnn, move=smove, mtype=smove.mtype))


def terminal_check(party1, party2):
    if party1.alive() and party2.alive():
        return False
    else: 
        return True
    

# This is the main function that simulates battle
def battle(wd, print_log = True):
    tline = wd.tline
    p1 = wd.party1
    p2 = wd.party2
    if print_log:
        print('Party 1: ', list(p1.dict.keys()))
        print('Party 2: ', list(p2.dict.keys()))
        print('------ Battle starts ------')
    
    init_atkr_event(p1, tline)
    init_atkr_event(p2, tline)
    log = []
    
    # execute events in order
    while not terminal_check(p1, p2):
        this_event = tline.pop()
        t = this_event.t
        event_name = this_event.name
        
        # case 1: a minion has a move freed
        if 'Free' in event_name:
            atkr = this_event.mnn_usedAtk
            if atkr.alive():
                move_to_free = this_event.move
                move_to_free.free_since = t
                player_AI_choose(wd, atkr, t, event_name)
        
        # case 2: a minion takes damage
        elif 'Hurt' in event_name:
            # update: dfdr's mp and hp, atkr's mp
            dmg_taker = this_event.mnn_hurt
            dmg_giver = this_event.mnn_usedAtk
            
            # case 2a: both atkr and dfdr alive when the move is being casted: 
            # the move is casted and causes damage
            if dmg_taker.alive() and dmg_giver.alive():
                dmg_taker.hp = max(dmg_taker.hp - this_event.damage, 0)
                if this_event.atkr_mpdelta < 0:
                    dmg_giver.mp = min(dmg_giver.mp + this_event.atkr_mpdelta, dmg_giver.mp_capacity)
            
            # case 2b: either atkr or dfdr dead before the move casted:
            # cost atkr's mp if it's an smove, but no damage, no mp increase 
            else: 
                this_event.damage = 0
                this_event.dfdr_mpdelta = 0
                if this_event.atkr_mpdelta < 0:
                    dmg_giver.mp = min(dmg_giver.mp + this_event.atkr_mpdelta, dmg_giver.mp_capacity)
                else:
                    this_event.atkr_mpdelta = 0
            
            # after the move casted, if a dfdr/atkr is alive, it gains mp
            if dmg_taker.alive():
                dmg_taker.mp = min(dmg_taker.mp + this_event.dfdr_mpdelta, dmg_taker.mp_capacity)
            if dmg_giver.alive() and this_event.atkr_mpdelta >= 0:
                dmg_giver.mp = min(dmg_giver.mp + this_event.atkr_mpdelta, dmg_giver.mp_capacity)
            
            # check if the dmg_taker can make an smove, since it gained mp
            if dmg_taker.alive():
                player_AI_choose(wd, dmg_taker, t, event_name)
            
            # check if the dmg_giver can make an smove if it gained mp
            if dmg_giver.alive() and this_event.atkr_mpdelta >= 0:
                player_AI_choose(wd, dmg_giver, t, event_name)
        
        log.append((t,this_event.name, this_event.msg))
        
        if print_log:
            hp_status = {}
            mp_status = {}
            for mid, mnn in p1.dict.items():
                hp_status[mid] = mnn.hp
                mp_status[mid] = mnn.mp
            for mid, mnn in p2.dict.items():
                hp_status[mid] = mnn.hp
                mp_status[mid] = mnn.mp
            print(t,this_event.name, this_event.msg)
            print('HP\n', hp_status)
            print('MP\n', mp_status)
            print('---------------------------------')
            
        
    # battle ends. assign winner and loser
    if p1.alive():
        winner = p1
        loser = p2
        if print_log: print('Party 1 wins')
    elif p2.alive():
        winner = p2
        loser = p1
        if print_log: print('Party 2 wins')
    
    winner_mid = list(winner.dict.keys())
    loser_mid = list(loser.dict.keys())
            
    return winner_mid, loser_mid, log


# demo
if __name__ == "__main__":

    # load data from json
    minion_data = load_minion_data('data/miniondata_1.json')
    skill_data = load_data('data/skilldata_1.json')
    
    # pick minions, unserialize dict to object
    mnn1 = BattleMinion(minion_dict=minion_data, skill_dict=skill_data, mid=1)
    mnn2 = BattleMinion(minion_dict=minion_data, skill_dict=skill_data, mid=2)
    
    # assign minions to parties
    party1 = Party([mnn1])
    party2 = Party([mnn2])

    wd1_params = {
        'minion_data':minion_data,
        'skill_data': skill_data,
        'party1': party1,
        'party2': party2,
        'battle_type': '1v1',
        'timelimit': 600,
        'tline': Timeline()
    }

    wd1 = World(**wd1_params)
    
    _,_,_ = battle(wd1)
