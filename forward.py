import logging

from forest_allocation import RandomForestAllocation
from src.reward import get_rewards
from src.simulator import Simulator
from src.misc import call_allocation_algorithm
from src.pools import generate_assets_and_pools


def query_and_score(allocations, assets_and_pools):
    simulator = Simulator()
    simulator.initialize()
    if assets_and_pools is not None:
        simulator.init_data(init_assets_and_pools=assets_and_pools)
    else:
        simulator.init_data()

    apys, max_apy = get_rewards(
        simulator,
        allocations_list=allocations,
    )

    logging.debug(f"apys: {apys}")
    logging.debug(f"max_apy:{max_apy}")
    return apys, max_apy


def calc_simple_allocations(assets_and_pools):
    pools = assets_and_pools['pools']
    total_asset = assets_and_pools['total_assets']
    simple_allocations = {k: total_asset / len(pools) for k, v in pools.items()}
    return simple_allocations


model = RandomForestAllocation()


def main():
    assets_and_pools = generate_assets_and_pools()

    simple_allocations = calc_simple_allocations(assets_and_pools)
    predicted_allocated = model.predict_allocation(assets_and_pools)
    dn_allocations = call_allocation_algorithm(assets_and_pools)

    apys, max_apy = query_and_score([simple_allocations, predicted_allocated, dn_allocations], assets_and_pools)
    logging.info(f"apys = {apys}")
    logging.info(f"max_max_max_apy: {max_apy}")
    return apys


if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    main()
