# `battle_simulator`    

This module implements player-vs-player battle simulations. Here's a simpe demo.    

Demo: *battle_simulation_demo.ipynb*
Input data sample: */data*    
    

## Game Mechanism    
------
A minion has **7 STATS**, which determine **4 SUBSTATS** and **3 ATTRIBUTES**.    
For detailed information, please refer to Lexicon & Definition.    

<img src="/Users/sibylhe/Documents/Minions/battle-simulation/img/minion_relationship_keys.jpg" style="zoom: 50%;" />

### Data Input and Preprocessing    
------
The simulator takes two inputs in json:    
- minion_data
- skill_data
  

<img src="/Users/sibylhe/Documents/Minions/battle-simulation/img/minion_relationship_keys_variable.jpg" style="zoom: 50%;" />

To create a **minion** instance, specify:    
id: unique id of the minion, `int`    
name: name of the minion, `string`    
base_stats: 1\*7 `array`, 7 STATS: strength, endurance, dexterity, obedience, constituition, intelligence, wisdom    
base_attributes: 1\*3 `array`, 3 ATTRIBUTES: hp, ap, ep    
stats_coeff: 2\*7 `array`, [[initial STATS], [coefficient]]. Linear relationship:  [SUBSTATS, ATTRIBUTES_CAPACITY] = [STATS] * [coefficient] + [initial STATS]  
mp_capacity: maximum mp, `int`    
mp_regen_per_second: # of mp the minion gains per second, `int`    
mp_gain_rate_damage_dealt: the rate of how much damage is transformed to mp, `float` (0, 1)    
mp_gain_rate_damage_taken: the rate of how much damage taken is transformed to mp, `float` (0, 1)    
wellness_factor: a multiplier on STATS, `float` [0.7, 1.3]    
fmoves: `list` of fast move id    
smoves: `list` of super move id     
    

#### `compute_current_status()` Function     
**Args**:    
minion `dict`    
**Returns**:    
minion `dict` 

Modifies the minion `dict` by computing `base_substats`, `cur_substats`, `cur_attributes`, and `cp`. This function is wrapped into `load_minion_data()`.       

**Current CP (combat power) formula**
$cp = strength \times \sqrt{endurance} \times \sqrt{constitution}$    
    

#### `load_minion_data()` Function     
**Args**:    
fn: file name    
update_luck: `bool`, default True    
update_stats_coeff: `bool`, default False    
**Returns**:    
minion_data `dict`    
    

#### `load_skill_data()` Function     
**Args**:    
fn: file name    
**Returns**:    
skill_data `dict`    
    
    

### Battle Classes    
------
####  `Skill` Class    
Initiated from skill_data `dict`.     

For fast moves, calculate **dps** (damage per second).    
$dps = \dfrac{power}{cd+cast}$    

For super moves, calculate **dpm** (damage per meter point).    
$dpm = \dfrac{power}{(cd + cast)\times|mpdelta|}$    
    

#### `BattleMinion` Class     
Initiated from minion_data `dict` and skill_data `dict`, with only battle-related attributes.
```
mnn1 = BattleMinion(minion_dict=minion_data, skill_dict=skill_data, mid=1)
```



#### `Party` Class     

Team. Initiated from a `list` of BattleMinion.
```
party1 = Party([mnn1])
```



#### `Event` Class    

Events sit on the timeline and carry instructions about what they should do.    
Adding events does *not* directly change BattleMinion `object`. The BattleMinion's attributes will not be updated until the event is executed.    

**Type of Event**    

1. **mnn_fCD**, **mnn_sCD**: Attacking minion used a move and the move starts cooldown. 

   update: move.free_since = t + move.cd

   takes (name, t, mnn_usedAtk, move, mtype)

2. **mnn_fFree**, **mnn_sFree**: Attacking minion is free to perform a fmove/smove.    
	takes (name, t, mnn_usedAtk, move, mtype)    
3. **mnnHurt**: Attacking minion makes damage and gains mp (fmove) or loses mp (smove). Defending minion takes damage and gains mp.

   takes (name, t, mnn_usedAtk, mnn_hurt, damage, atkr_mpdelta, dfdr_mpdelta, move, mtype)    
       

#### `Timeline` Class    
Timeline is an ordered queue of events.    

| t    | event     | mnn_usedAtk | mnn_hurt | move    | damage | atkr_mpdelta | dfdr_mpdelta |
| ---- | --------- | ----------- | -------- | ------- | ------ | ------------ | ------------ |
| 0    | mnn_fFree | mnn1        | None     | fmove_1 | None   | None         | None         |
| 1    | mnnHurt   | mnn1        | mnn2     | fmove_1 | 130    | 26           | 30           |

​    

#### `World` Class    
Holds all settings. Initiated from kwargs: minion_data, skill_data, party1, party2, battle_type, timelimit, tline
```
wd_params = {
    'minion_data':minion_data,
    'skill_data': skill_data,
    'party1': party1,
    'party2': party2,
    'battle_type': '1v1',
    'timelimit': 600,
    'tline': Timeline()
}
wd = World(**wd_params)
```



### Battle Functions    
------
#### `compute_damage()` Function    
**Args**:    
atkr: BattleMinion `object`    
dfdr: BattleMinion `object`    
move: Skill `object`    
**Returns**:    
damage: `int`    
atkr_mpdelta: how many mp the attacker gains/loses    
dfdr_mpdelta: how many mp the defender gains    

**Current damage formula**

$damage = \dfrac{atkr.attack}{dfdr.defense} \times \textit{critical multiplier} \times move.power \times \textit{hit multiplier}$     

$\textit{critical multiplier}$: $2$ if this is a critical hit, $1$ if it is not (determined by dexterity)    
$\textit{hit multiplier}$: $1$ if the attacking minion successfully hits the defending minion, $0$ if it misses (determined by obedience)    
    
    

#### `mnn_use_move()` Function    
**Args**:    
mnn_usedAtk: BattleMinion `object`    
mnn_hurt: BattleMinion `object`    
move: Skill `object`    
wd: World `object`    
t: current time, `float`    
    
This is equivalent to a player pressing "Attack" button, will add 3 events to timeline:    

- t: mnn_**x**CD    
- t + move.cast: mnnHurt    
- t + move.cd: mnn_**x**Free    
  
    
### Battle Simulation Functions    
------
#### `player_AI_choose()` Function    
**Args**:    
wd: World `object`    
atkr: BattleMinion `object`    
t: current time, `float`    
event_name: `string`    

This function is the strategy of the attacking minion at its turn. It manipulates the timeline by inserting events, will call `mnn_use_move()` if a best move is found.
    
**Logic**    
available_moves = {}    
**case 1**: “mnn_sFree” or “mnnHurt”    
If minion’s current mp >= min mp threshold of its super moves (super move possible):    
`search_smoves()`:    
for smove in smoveset:    
    if cd cleared (free_since <= t) and enough mp (mpdelta + atkr.mp >= 0):       
        available_moves[smove.sid] = smove.dpm    
    **case 1a**: 2+ super moves available: sort by dpm = power/(cd+cast)/abs(mpdelta)    
    **case 1b**: 1 super move available: use it    
    **case 1c**: 0 super moves available: do nothing    
    
**case 2**: “mnn_fFree”    
`search_fmoves()`:    
for fmove in fmoveset:
    if cd cleared (free_since <= t):    
        available_moves[fmove.sid] = fmove.dps    
    **case 2a**: 2+ fast moves available: sort by dps = power/(cd+cast)    
    **case 2b**: 1 fast move available: use it    
    **case 2c**: 0 fast moves available: do nothing    
    
If a best move is found:    
select a defending minion to attack (current strategy: minion with lowest hp in opponent's team):    
    dfdr  = `select_dfdr()`    
    `mnn_use_move(atkr, dfdr, move, mtype, wd, t)`    
    
    

#### `battle()` Function    
**Args**:    
wd: World `object`    
print_log: `bool`, if True, will print the log to the console    
**Returns**:    
winner_mid: ids of winning minions      
loser_mid: ids of the losing minions    
log: records of the battle, `list` of (time, event name, message)    

This function simulates a battle, will loop through the timeline and execute the events in order, until one party dies.    
    
**Logic**    

**STEP 1**: initiate a battle
for all minions:
	free all moves

**STEP 2**: execute events in order 

```
this_event = tline.pop()
t = this_event.t
event_name = this_event.name
```

**case 1**: "mnn_**x**CD"
a minion used a move, the move starts cooldown
update: move.free_since = t + move.cd

**case 2**: "mnn_**x**Free"
a minion has a move freed, search corresponding moveset and choose one move
`player_AI_choose(wd, atkr, t, event_name)`

**case 3**: "mnnHurt"
a minion takes damage
update: dfdr's mp and hp, atkr's mp

check if the dmg_taker can make an smove, since it gained mp:
`player_AI_choose(wd, dmg_taker, t, event_name)`

​	check if the dmg_giver can make an smove if it gained mp:
​		`player_AI_choose(wd, dmg_giver, t, event_name)`

**STEP 3**: assign winner and loser