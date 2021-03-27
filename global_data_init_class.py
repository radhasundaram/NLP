#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on 05/06/20 2:42 AM
# @author: Gurusankar G

from __future__ import division

import ast
import json
import logging
import re

import numpy as np

from ta_lib.core import dataset
from ta_lib.core.api import create_context, initialize_environment
from ta_lib.core.utils import get_package_path


def get_pack_path():
    """Absolute path for the package."""
    return get_package_path().replace("\\", "/").replace("src", "")


initialize_environment(debug=False, hide_warnings=True)
config_path = get_pack_path() + "notebooks/tpo/python/conf/config.yml"  # noqa
context = create_context(config_path)
logger = logging.getLogger(__name__)


class global_initializer:
    """Usage : The class is the main input format for the three stage optimizer.

    The input required is the retailer number and category numer to be
    created with the class construct
    ex - global_initializer(Retailer, Category)found
    in the Caller.py file. The class is internally
    used by all the three phases of the solver to save
    their interim and final values
    """

    def __init__(self, combo_data):

        # Config constants initialization complete ######

        def formatter(data, conv):
            if conv == "int":
                return int(data)
            elif conv == "float":
                return float(data)
            elif conv == "list":
                return json.loads(data)
            elif conv == "dict":
                return ast.literal_eval(data)
            else:
                return data

        self.Event_Buffer = formatter(combo_data["event_buffer"], conv="int")
        self.EDLP_Perc_Limit = formatter(combo_data["edlp_perc_limit"], conv="int")
        self.TPR_Perc_Limit = formatter(combo_data["tpr_perc_limit"], conv="int")
        self.Ov_Perc_Limit = formatter(combo_data["overall_perc_limit"], conv="int")
        self.Total_EDLP_Perc_Limit = formatter(
            combo_data["total_edlp_perc_limit"], conv="int"
        )
        self.Total_TPR_Perc_Limit = formatter(
            combo_data["total_tpr_perc_limit"], conv="int"
        )
        self.EDLP_LB = formatter(combo_data["edlp_lb"], conv="float")
        self.EDLP_UB = formatter(combo_data["edlp_ub"], conv="float")
        self.TPR_LB = formatter(combo_data["tpr_lb"], conv="float")
        self.TPR_UB = formatter(combo_data["tpr_ub"], conv="float")
        self.Retailer_Overall_Sales_Buffer = formatter(
            combo_data["retailer_overall_sales_buffer"], conv="int"
        )

        self.Weeks_Baseline_2 = formatter(
            combo_data["constraint_set_st2"]["Weeks_Baseline_2"], conv="list"
        )
        self.Same_Cal_PPG_2 = formatter(
            combo_data["constraint_set_st2"]["Same_Cal_PPG_2"], conv="list"
        )

        self.LB_UB_Limits_3 = formatter(
            combo_data["constraint_set_st3"]["LB_UB_Limits_3"], conv="dict"
        )
        self.Promo_Wk_Limit_3 = formatter(
            combo_data["constraint_set_st3"]["Promo_Wk_Limit_3"], conv="list"
        )
        self.Promo_rep_Wk_Limit_3 = formatter(
            combo_data["constraint_set_st3"]["Promo_rep_Wk_Limit_3"], conv="dict"
        )
        self.Deep_Promos_Collective_3 = formatter(
            combo_data["constraint_set_st3"]["Deep_Promos_Collective_3"], conv="dict"
        )
        self.DPC_nweeks_3 = formatter(
            combo_data["constraint_set_st3"]["DPC_nweeks_3"], conv="list"
        )
        self.Promos_Rep_PR_3 = formatter(
            combo_data["constraint_set_st3"]["Promos_Rep_PR_3"], conv="list"
        )
        self.PR_nweeks_3 = formatter(
            combo_data["constraint_set_st3"]["PR_nweeks_3"], conv="list"
        )
        self.Weeks_Baseline_3 = formatter(
            combo_data["constraint_set_st3"]["Weeks_Baseline_3"], conv="list"
        )
        self.Promos_Nos_3 = formatter(
            combo_data["constraint_set_st3"]["Promos_Nos_3"], conv="list"
        )
        self.RSV_Cons_val_3 = formatter(
            combo_data["constraint_set_st3"]["RSV_Cons_val_3"], conv="list"
        )
        self.Clash_PPG_3 = formatter(
            combo_data["constraint_set_st3"]["Clash_PPG_3"], conv="list"
        )
        self.Same_Cal_PPG_3 = formatter(
            combo_data["constraint_set_st3"]["Same_Cal_PPG_3"], conv="list"
        )
        self.Deep_Act_3 = formatter(
            combo_data["constraint_set_st3"]["Deep_Act_3"], conv="dict"
        )

        self.TIME_STAGE2 = formatter(combo_data["time_step_2"], conv="int")
        self.TIME_STAGE3 = formatter(combo_data["time_step_3"], conv="int")
        self.TIME_SINGLESHOT = formatter(combo_data["time_step_single"], conv="int")

        # ###### GLOBALS name definition ########
        self.Ret = int(combo_data["retailer"])
        self.Cat = int(combo_data["category"])
        self.stage1_EDLP_Init_PPG = None
        self.stage1_TPR_Init_PPG = None
        self.stage1_Num_Events_PPG = None
        self.stage1_EDLP_Spend_Values = None
        self.stage1_TPR_Spend_Values = None
        self.stage1_Tot_Spend_Values = None
        self.FLAG_Inter = None
        self.EDLP_Inter = None
        self.TPR_Inter = None
        self.Solver_Model_Result = None
        self.EDLP_Final = None
        self.TPR_Final = None

        self.Stage1Success = False
        self.Stage2Success = False
        self.Stage3Success = False

    def class_loader(self):
        """Load historical data. coefficients and Base values."""
        try:
            self.load_retailer_category()
            self.prepare_data()
            return 1
        except Exception as e:
            logger.error("\n\nError in loading data" + str(e))
            return 0

    def load_retailer_category(self):
        """Load Historical data. coefficients Helper Function1."""
        Ret = self.Ret
        Cat = self.Cat

        annual_df = dataset.load_dataset(context, "TPO_Input/Final_Spend_Data")
        annual_df = annual_df[
            (annual_df.Retailer == "Retailer" + str(Ret))
            & (annual_df.Category == "Category" + str(Cat))
        ].reset_index(drop=True)
        coef_df = dataset.load_dataset(
            context, "TPO_Input/Coeff_Matrix", Ret=str(Ret), Cat=str(Cat)
        )  # noqa
        Tot_Prod = coef_df["Product"].nunique()
        Tot_Week = coef_df["wk"].nunique()

        quarter_df = dataset.load_dataset(context, "TPO_Input/Final_Base_Price")
        quarter_df = quarter_df[
            (quarter_df.Retailer == "Retailer" + str(Ret))
            & (quarter_df.Category == "Category" + str(Cat))
        ].reset_index(drop=True)
        historical_df = dataset.load_dataset(context, "TPO_Input/Historical_Events")
        historical_df = historical_df[
            (historical_df.Retailer == "Retailer" + str(Ret))
            & (historical_df.Category == "Category" + str(Cat))
        ].reset_index(drop=True)

        Baseline_Promo = np.array(historical_df["Promo_Events"]).reshape(
            Tot_Prod, Tot_Week
        )

        Baseline_FLAG = np.array(historical_df["Event"]).reshape(Tot_Prod, Tot_Week)

        TPR_promo_data_df = dataset.load_dataset(context, "TPO_Input/TPR_Perc_Val")

        TPR_promo_data_df = TPR_promo_data_df[
            (TPR_promo_data_df.Retailer == "Retailer" + str(Ret))
            & (TPR_promo_data_df.Category == "Category" + str(Cat))
        ].reset_index(drop=True)

        TPR_promo_data_df = TPR_promo_data_df[TPR_promo_data_df.Promo_Map != 0]

        TPR_promo_data_df.sort_values(
            by=["Product", "Promo_Map"], ignore_index=True, inplace=True
        )

        TPR_Prd_Grp = TPR_promo_data_df.groupby("Product")

        Prds = TPR_Prd_Grp.count().index

        TPR_Perc_Val = []

        for i in range(len(Prds)):
            TPR_Perc_Val.append(
                list(TPR_Prd_Grp.get_group(Prds[i])["TPR_Percentage_Value"])
            )

        Cat_Display = dataset.load_dataset(context, "TPO_Input/Cat_Disp_File")

        Cat_Display = Cat_Display[
            (Cat_Display.Retailer == "Retailer" + str(Ret))
            & (Cat_Display.Category == "Category" + str(Cat))
        ].reset_index(drop=True)

        Catalogue = np.array(Cat_Display["Catalogue"]).reshape(Tot_Prod, Tot_Week)

        Display = np.array(Cat_Display["Display"]).reshape(Tot_Prod, Tot_Week)

        Cat_Sum = [sum(i) for i in Catalogue]

        Disp_Sum = [sum(i) for i in Display]

        TE_Coeffs_df = dataset.load_dataset(context, "TPO_Input/TE_Val")

        TE_Coeffs_df = TE_Coeffs_df[
            (TE_Coeffs_df.Retailer == "Retailer" + str(Ret))
            & (TE_Coeffs_df.Category == "Category" + str(Cat))
        ].reset_index(drop=True)

        # TE_Coeffs_df=TE_Coeffs_df[TE_Coeffs_df.Promo_Map!=0]

        TE_Coeffs_df.sort_values(
            by=["Product", "Promo_Map"], ignore_index=True, inplace=True
        )

        TE_Prd_Grp = TE_Coeffs_df.groupby("Product")

        TE_Val = []

        for i in range(len(Prds)):
            TE_Val.append(list(TE_Prd_Grp.get_group(Prds[i])["TE"]))
        TPR_Cutoff = dataset.load_dataset(context, "TPO_Input/TPR_Cutoff")

        TPR_Cutoff = TPR_Cutoff[
            (TPR_Cutoff.Retailer == "Retailer" + str(Ret))
            & (TPR_Cutoff.Category == "Category" + str(Cat))
        ].reset_index(drop=True)

        self.annual_df = annual_df
        self.coef_df = coef_df
        self.quarter_df = quarter_df
        self.historical_df = historical_df
        self.TPR_promo_data_df = TPR_promo_data_df
        self.TE_Coeffs_df = TE_Coeffs_df
        self.TPR_Cutoff = TPR_Cutoff
        self.TPR_Perc_Val = TPR_Perc_Val
        self.TE_Prd_Grp = TE_Prd_Grp
        self.TE_Val = TE_Val
        self.Baseline_Promo = Baseline_Promo
        self.Baseline_FLAG = Baseline_FLAG
        self.Catalogue = Catalogue
        self.Display = Display
        self.annual_df = annual_df
        self.Disp_Sum = Disp_Sum
        self.Cat_Sum = Cat_Sum
        self.historical_df = historical_df

    def prepare_data(self):
        """Load Historical data. coefficients Helper Function2."""
        annual_df = self.annual_df
        coef_df = self.coef_df
        quarter_df = self.quarter_df
        #         historical_df = self.historical_df
        Event_Buffer = self.Event_Buffer

        Tot_Prod = coef_df["Product"].nunique()
        # Tot_Week = coef_df["wk"].nunique()
        Tot_Week = 52

        EDLP_Events = list(annual_df["RP_Events"])
        Min_EDLP_Events = [
            i - Event_Buffer if i - Event_Buffer >= 0 else 0 for i in EDLP_Events
        ]
        Max_EDLP_Events = [
            i + Event_Buffer if i + Event_Buffer < Tot_Week + 1 else Tot_Week
            for i in EDLP_Events
        ]

        TPR_Events = list(annual_df["TPR_Events"])
        Min_TPR_Events = [
            i - Event_Buffer if i - Event_Buffer >= 0 else 0 for i in TPR_Events
        ]
        Max_TPR_Events = [
            i + Event_Buffer if i + Event_Buffer < Tot_Week + 1 else Tot_Week
            for i in TPR_Events
        ]

        Target_EDLP_Spend = [i for i in annual_df["PPG_RP_Spend"]]
        Target_TPR_Spend = [i for i in annual_df["PPG_TPR_Spend"]]
        Target_Trade_Spend = [i for i in annual_df["PPG_Total_Spend"]]

        Mapping = {}
        Prod_Ind = coef_df["Product"][0:Tot_Prod]
        for i, j in zip(Prod_Ind.index, Prod_Ind.values):
            Mapping[j] = i
        Mapping_reverse = {i: j for j, i in Mapping.items()}

        constants = [i for i in coef_df["constant"]]

        Cat_Coef = coef_df["Catalogue"][0:Tot_Prod]

        Disp_Coef = coef_df["Display"][0:Tot_Prod]

        Base_Price_stg1 = [i for i in quarter_df["Final_baseprice"]]
        Intercepts_stg1 = []
        for pr in range(Tot_Prod):
            Intercepts_stg1.append(
                np.mean([constants[j * Tot_Prod + pr] for j in range(0, Tot_Week)])
            )

        Base_Price_stg2 = [[i] * Tot_Week for i in quarter_df["Final_baseprice"]]
        Intercepts_stg2 = [
            constants[j : j + Tot_Prod] for j in range(0, len(constants), Tot_Prod)
        ]  # noqa

        EDLP_Coef = np.array(
            coef_df[[i for i in coef_df.columns if i.count("Retailer_Regular") == 1]]
        )
        TPR_Coef = np.array(
            coef_df[[i for i in coef_df.columns if i.count("Retailer_Promoted") == 1]]
        )

        # ################################ Available EDLP Interactions pairs ##############################

        EDLP = [
            re.findall(r"[0-9]+", i)
            for i in coef_df.columns
            if i.count("Retailer_Regular") > 1
        ]
        EDLP_Interactions = []
        for i in EDLP:
            temp = []
            for j in i:
                temp.append(int(j))
            EDLP_Interactions.append(temp)

        # ###################################### Available TPR Interactions pairs #########################

        TPR = [
            re.findall(r"[0-9]+", i)
            for i in coef_df.columns
            if i.count("Retailer_Promoted") > 1
        ]
        TPR_Interactions = []
        for i in TPR:
            temp = []
            for j in i:
                temp.append(int(j))
            TPR_Interactions.append(temp)

        # ###################################### EDLP_Interaction_Coef_Values ############################

        EDLP_Int_Coef_Values = {}
        for col in coef_df.columns:
            if col.count("Retailer_Regular") > 1:
                Pair_name = "_".join([str(int(i)) for i in re.findall(r"[0-9]+", col)])
                EDLP_Int_Coef_Values[Pair_name] = list(coef_df[col])

        # ###################################### TPR_Interaction_Coef_Values #############################

        TPR_Int_Coef_Values = {}
        for col in coef_df.columns:
            if col.count("Retailer_Promoted") > 1:
                Pair_name = "_".join([str(int(i)) for i in re.findall(r"[0-9]+", col)])
                TPR_Int_Coef_Values[Pair_name] = list(coef_df[col])

        # ##################################### Loading Pantry Loading Coefficients #######################

        Pantry_1 = list(coef_df["Pantry_Loading_1"])
        Pantry_1 = [
            Pantry_1[j : j + Tot_Prod] for j in range(0, len(Pantry_1), Tot_Prod)
        ]
        Pantry_2 = list(coef_df["Pantry_Loading_2"])
        Pantry_2 = [
            Pantry_2[j : j + Tot_Prod] for j in range(0, len(Pantry_2), Tot_Prod)
        ]

        #     TE_Coeff = np.array(Promo_df[["TE_Promo","TE_NoPromo"]])
        self.Tot_Prod = Tot_Prod
        self.Tot_Week = Tot_Week
        self.EDLP_Events = EDLP_Events
        self.Min_EDLP_Events = Min_EDLP_Events
        self.Max_EDLP_Events = Max_EDLP_Events
        self.TPR_Events = TPR_Events
        self.Min_TPR_Events = Min_TPR_Events
        self.Max_TPR_Events = Max_TPR_Events

        self.Target_EDLP_Spend = Target_EDLP_Spend
        self.Target_TPR_Spend = Target_TPR_Spend
        self.Target_Trade_Spend = Target_Trade_Spend
        self.Mapping = Mapping
        self.Mapping_reverse = Mapping_reverse
        self.constants = constants
        self.EDLP_Coef = EDLP_Coef
        self.TPR_Coef = TPR_Coef

        self.EDLP_Interactions = EDLP_Interactions
        self.TPR_Interactions = TPR_Interactions
        self.EDLP_Int_Coef_Values = EDLP_Int_Coef_Values
        self.TPR_Int_Coef_Values = TPR_Int_Coef_Values
        self.Pantry_1 = Pantry_1
        self.Pantry_2 = Pantry_2

        self.Base_Price_stg1 = Base_Price_stg1
        self.Intercepts_stg1 = Intercepts_stg1
        self.Base_Price_stg2 = Base_Price_stg2
        self.Intercepts_stg2 = Intercepts_stg2

        self.Cat_Coef = Cat_Coef
        self.Disp_Coef = Disp_Coef
