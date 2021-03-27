#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on 05/06/20 3:55 PM
# @author: Gurusankar G

import collections
import datetime
import itertools
import logging
import time

import xmltodict
from tqdm import tqdm

# Necessary imports includes importing Triple optimizer package
import ta_lib.tpo.optimization.discrete.results_helper as res
from ta_lib.core.api import create_context, initialize_environment
from ta_lib.core.utils import configure_logger, get_package_path
from ta_lib.tpo.optimization.discrete.global_data_init_class import (  # noqa
    global_initializer,
)
from ta_lib.tpo.optimization.discrete.stage_3_tpo_optimizer import Stage3_caller


def get_pack_path():
    """Absolute path for the package."""
    return get_package_path().replace("\\", "/") + "/"


def current_time():
    return datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")


initialize_environment(debug=False, hide_warnings=True)
config_path = (
    get_pack_path() + "../notebooks/tpo/python/conf/config.yml"
)  # noqa

context = create_context(config_path)

logger = logging.getLogger(__name__)
log_path = get_pack_path() + "../logs/tpo_optimizer_" + current_time() + ".log"
logger = configure_logger(log_file=log_path, log_level="DEBUG")

# Inputs for Particular Retailer Category Combination is stored int input.xml

# Reading and making the relevant information for
# each retailer and category stored in input.xml
optim_config_path = (
    get_pack_path()
    + "../notebooks/tpo/python/conf/Discrete_optimizer_config.xml"
)

with open(optim_config_path) as fd:  # noqa
    doc = xmltodict.parse(fd.read())
head = doc["data"]
Retailer_Category_Combo = head["ret_cat_combo"]

if isinstance(Retailer_Category_Combo, collections.OrderedDict):
    Retailer_Category_Combo = [Retailer_Category_Combo]

if __name__ == "__main__":

    for single_combo in tqdm(Retailer_Category_Combo):
        global Globals
        # calling the global initializer for creating a concrete class
        # for global data availability across optimizer stages
        Globals = global_initializer(single_combo)
        check = globals
        counter = 0  # track how many combinations converge
        combo = [False, True]
        z = 0
        const_comb = list(itertools.product(combo, repeat=13))[1:]

        for single_comb in const_comb:

            single_comb = [
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                True,
                True,
                True,
                False,
                False,
                False,
            ]
            sec_sing_comb = [
                single_comb[0],
                single_comb[6],
                single_comb[11],
                single_comb[12],
            ]

            if Globals.class_loader():

                s = time.time()
                Stage3_caller(Globals, single_comb, sec_sing_comb)
                e = time.time()
                if (
                    Globals.Stage3Success
                    & Globals.Stage2Success
                    & Globals.Stage1Success
                ):
                    St1 = Globals.Stage1Success
                    St2 = Globals.Stage2Success
                    St3 = Globals.Stage3Success

                    if St1 & St2 & St3:

                        res.save_results(Globals, runtime=e - s)
            break
