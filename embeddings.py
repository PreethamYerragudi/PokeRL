import numpy as np

from gymnasium.spaces import Box
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.data import GenData

EMBED_DICT = {0: 'ex_embedding'}

def select_embedding(embedding: str):
    if embedding == 'ex_embedding':
        return _ex_embedding, _ex_embedding_description
    else:
        print("Invalid Embedding")
        exit()

def _ex_embedding(battle: AbstractBattle, gen_data: GenData):
    # -1 indicates that the move does not have a base power
    # or is not available
    moves_base_power = -np.ones(4)
    moves_dmg_multiplier = np.ones(4)
    for i, move in enumerate(battle.available_moves):
        moves_base_power[i] = (
            move.base_power / 100
        )  # Simple rescaling to facilitate learning
        if move.type:
            moves_dmg_multiplier[i] = move.type.damage_multiplier(
                type_1=battle.opponent_active_pokemon.type_1,
                type_2=battle.opponent_active_pokemon.type_2,
                type_chart=gen_data.type_chart
            )
    # We count how many pokemons have fainted in each team
    fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
    fainted_mon_opponent = (
        len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
    )
    # Final vector with 10 components
    final_vector = np.concatenate(
        [
            moves_base_power,
            moves_dmg_multiplier,
            [fainted_mon_team, fainted_mon_opponent],
        ]
    )
    return np.float32(final_vector)

def _ex_embedding_description():
    low = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
    high = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1]
    return Box(
        np.array(low, dtype=np.float32),
        np.array(high, dtype=np.float32),
        dtype=np.float32
    )