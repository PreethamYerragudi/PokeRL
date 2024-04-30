from poke_env.environment import AbstractBattle

REW_DICT = {0: "simple"}

def select_reward(reward: str):
    if reward == 'simple':
        return simple_reward
    else:
        print("Invalid Reward Type")
        exit()
    
def simple_reward(last_battle: AbstractBattle, battle: AbstractBattle, params: tuple):
    fainted_value, hp_value, number_of_pokemon, status_value, victory_value = params

    current_value = 0
    for mon in battle.team.values():
        current_value += mon.current_hp_fraction * hp_value
        if mon.fainted:
            current_value -= fainted_value
        elif mon.status is not None:
            current_value -= status_value

    for mon in battle.opponent_team.values():
        current_value -= mon.current_hp_fraction * hp_value
        if mon.fainted:
            current_value += fainted_value
        elif mon.status is not None:
            current_value += status_value

    if battle.won:
        current_value += victory_value
    elif battle.lost:
        current_value -= victory_value

    return current_value