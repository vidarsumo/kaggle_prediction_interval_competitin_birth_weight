# Prediction interval competition I: Birth weight



# 1.0.0 Setup -------------------------------------------------------------

library(tidyverse)
library(tidymodels)
library(finetune)
library(probably)

# For LightGBM and Catboost
library(bonsai)
library(treesnip)


# 1.1.0 Data --------------------------------------------------------------

data_raw_tbl <- read_csv("00_data/train.csv") %>% 
  janitor::clean_names()

test_set_data_tbl <- read_csv("00_data/test.csv") %>% 
  janitor::clean_names()

# 1.2.0 Data preparation --------------------------------------------------

col_names <- names(data_raw_tbl)

data_raw_tbl %>%
  select_if(is.numeric) %>% 
  select(-id) %>% 
  pivot_longer(cols = everything()) %>% 
  filter(name %in% col_names[31:40]) %>% 
  ggplot(aes(x = value)) + 
  geom_histogram() + 
  facet_wrap(~name, scales = "free")



# 1.2.1 Recode missing values ---------------------------------------------
# See info here: https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/DVS/natality/UserGuide2018-508.pdf

# Train data ----
data_tbl <- data_raw_tbl %>% 
  mutate(
    attend      = if_else(attend == 9, NA_real_, attend),
    bfacil      = if_else(bfacil == 9, NA_real_, bfacil),
    bmi         = if_else(bmi == 99.9, NA_real_, bmi),
    cig_0       = if_else(cig_0 == 99, NA_real_, cig_0),
    dlmp_mm     = if_else(dlmp_mm == 99, NA_real_, dlmp_mm),
    dmar        = if_else(dmar == 9, NA_real_, dmar),
    dob_tt      = if_else(dob_tt == 9999, NA_real_, dob_tt),
    fagecomb    = if_else(fagecomb == 99, NA_real_, fagecomb),
    feduc       = if_else(feduc == 9, NA_real_, feduc),
    illb_r      = if_else(illb_r == 999, NA_real_, illb_r),
    ilop_r      = if_else(ilop_r == 999, NA_real_, ilop_r),
    ilp_r       = if_else(ilp_r == 999, NA_real_, ilp_r),
    mbstate_rec = if_else(mbstate_rec == 3, NA_real_, mbstate_rec),
    meduc       = if_else(meduc == 9, NA_real_, meduc),
    m_ht_in     = if_else(m_ht_in == 99, NA_real_, m_ht_in),
    no_infec    = if_else(no_infec == 9, NA_real_, no_infec),
    no_mmorb    = if_else(no_mmorb == 9, NA_real_, no_mmorb),
    no_risks    = if_else(no_risks == 9, NA_real_, no_risks),
    pay         = if_else(pay == 9, NA_real_, pay),
    precare     = if_else(precare == 99, NA_real_, precare),
    previs      = if_else(previs == 99, NA_real_, previs),
    priordead   = if_else(priordead == 99, NA_real_, priordead),
    priorlive   = if_else( priorlive == 99, NA_real_,  priorlive),
    priorterm   = if_else(priorterm == 99, NA_real_, priorterm),
    p_wgt_r     = if_else(p_wgt_r == 999, NA_real_, p_wgt_r),
    rdmeth_rec  = if_else(rdmeth_rec == 9, NA_real_, rdmeth_rec),
    rf_cesarn   = if_else(rf_cesarn == 99, NA_real_, rf_cesarn),
    wtgain      = if_else(wtgain == 99, NA_real_, wtgain),
    dbwt        = if_else(dbwt == 9999, NA_real_, dbwt)
  )


# Some outcomes (dbwt might be missing, remove them)
data_tbl <- data_tbl %>% 
  drop_na(dbwt)


# Test data ----
test_set_data_tbl <- test_set_data_tbl %>% 
  mutate(
    attend      = if_else(attend == 9, NA_real_, attend),
    bfacil      = if_else(bfacil == 9, NA_real_, bfacil),
    bmi         = if_else(bmi == 99.9, NA_real_, bmi),
    cig_0       = if_else(cig_0 == 99, NA_real_, cig_0),
    dlmp_mm     = if_else(dlmp_mm == 99, NA_real_, dlmp_mm),
    dmar        = if_else(dmar == 9, NA_real_, dmar),
    dob_tt      = if_else(dob_tt == 9999, NA_real_, dob_tt),
    fagecomb    = if_else(fagecomb == 99, NA_real_, fagecomb),
    feduc       = if_else(feduc == 9, NA_real_, feduc),
    illb_r      = if_else(illb_r == 999, NA_real_, illb_r),
    ilop_r      = if_else(ilop_r == 999, NA_real_, ilop_r),
    ilp_r       = if_else(ilp_r == 999, NA_real_, ilp_r),
    mbstate_rec = if_else(mbstate_rec == 3, NA_real_, mbstate_rec),
    meduc       = if_else(meduc == 9, NA_real_, meduc),
    m_ht_in     = if_else(m_ht_in == 99, NA_real_, m_ht_in),
    no_infec    = if_else(no_infec == 9, NA_real_, no_infec),
    no_mmorb    = if_else(no_mmorb == 9, NA_real_, no_mmorb),
    no_risks    = if_else(no_risks == 9, NA_real_, no_risks),
    pay         = if_else(pay == 9, NA_real_, pay),
    precare     = if_else(precare == 99, NA_real_, precare),
    previs      = if_else(previs == 99, NA_real_, previs),
    priordead   = if_else(priordead == 99, NA_real_, priordead),
    priorlive   = if_else( priorlive == 99, NA_real_,  priorlive),
    priorterm   = if_else(priorterm == 99, NA_real_, priorterm),
    p_wgt_r     = if_else(p_wgt_r == 999, NA_real_, p_wgt_r),
    rdmeth_rec  = if_else(rdmeth_rec == 9, NA_real_, rdmeth_rec),
    rf_cesarn   = if_else(rf_cesarn == 99, NA_real_, rf_cesarn),
    wtgain      = if_else(wtgain == 99, NA_real_, wtgain)
  )


# 2.0.0 Modelling ---------------------------------------------------------

# Split data
splits <- data_tbl %>% initial_validation_split(prop = c(0.5, 0.25), strata = dbwt)

train_data_tbl <- splits %>% training()
val_data_tbl   <- splits %>% validation()
test_data_tbl  <- splits %>% testing()


# Use fraction of data to tune model
fraction_used <- 0.4
resamples <- sample_frac(train_data_tbl, size = fraction_used) %>% vfold_cv(v = 10)


# 2.1.0 Recipes -----------------------------------------------------------

# No steps
rec_base <- recipe(dbwt ~ ., data = train_data_tbl)

# Dummy encoding
rec_dummy <- rec_base %>% 
  step_dummy(all_nominal_predictors())



# 2.2.0 Single models - No stacking ---------------------------------------

ctrl_race <- control_race(verbose_elim = TRUE, allow_par = TRUE)


# 2.2.1 Catboost ----------------------------------------------------------

# * Special handling for parallel processing and Catboost
cl <- parallel::makeCluster(parallel::detectCores())

result <- parallel::clusterEvalQ(cl, {
  sink(file=NULL)
  library(tidymodels)
  library(catboost)
  library(treesnip)
  sink()
})

doParallel::registerDoParallel(cl)


cat_spec <- boost_tree(
  mode       = "regression",
  trees      = tune::tune(),
  min_n      = tune::tune(),
  tree_depth = tune::tune(),
  learn_rate = tune::tune()
  ) %>% 
  set_engine("catboost")


cat_wflw <- workflow() %>% 
  add_model(cat_spec) %>% 
  add_recipe(rec_base)


cat_tuned <- tune_race_anova(
  object     = cat_wflw,
  resamples  = resamples,
  param_info = parameters(cat_wflw),
  grid       = 100,
  metrics    = metric_set(rmse),
  control    = ctrl_race
  )


cat_fin <- finalize_workflow(cat_wflw, select_best(cat_tuned)) %>% 
  fit(train_data_tbl)


# * Stop cluster for Catboost
parallel::stopCluster(cl)
gc()




# * Start parallel for other algorithms -----------------------------------

cl <- parallel::makeCluster(parallel::detectCores())
doParallel::registerDoParallel(cl)


# 2.2.2 XGBoost -----------------------------------------------------------

xgboost_spec <- boost_tree(
  mode           = "regression",
  mtry           = tune::tune(),
  trees          = tune::tune(),
  min_n          = tune::tune(),
  tree_depth     = tune::tune(),
  learn_rate     = tune::tune(),
  loss_reduction = tune::tune(),
  sample_size    = tune::tune(),
  stop_iter      = tune::tune()
) %>% 
  set_engine("xgboost")


xgboost_wflw <- workflow() %>% 
  add_model(xgboost_spec) %>% 
  add_recipe(rec_dummy)


xgboost_tuned <- tune_race_anova(
  object     = xgboost_wflw,
  resamples  = resamples,
  param_info = parameters(xgboost_wflw),
  grid       = 100,
  metrics    = metric_set(rmse),
  control    = ctrl_race
)


xgboost_fin <- finalize_workflow(xgboost_wflw, select_best(xgboost_tuned)) %>% 
  fit(train_data_tbl)



# 2.2.3 LightGBM ----------------------------------------------------------

lightgbm_spec <- boost_tree(
  mode           = "regression",
  mtry           = tune::tune(),
  trees          = tune::tune(),
  min_n          = tune::tune(),
  tree_depth     = tune::tune(),
  learn_rate     = tune::tune(),
  loss_reduction = tune::tune()
  ) %>% 
  set_engine("lightgbm")


lightgbm_wflw <- workflow() %>% 
  add_model(lightgbm_spec) %>% 
  add_recipe(rec_base)


lightgbm_tuned <- tune_race_anova(
  object     = lightgbm_wflw,
  resamples  = resamples,
  param_info = parameters(lightgbm_wflw),
  grid       = 100,
  metrics    = metric_set(rmse),
  control    = ctrl_race
)


lightgbm_fin <- finalize_workflow(lightgbm_wflw, select_best(lightgbm_tuned)) %>% 
  fit(train_data_tbl)




# * Stop parallel for other algorithms ------------------------------------

foreach::registerDoSEQ()
gc()



# 2.4.0 Intervals ---------------------------------------------------------


# 2.4.1 Create resamples using best model parameters
ctrl  <- control_resamples(save_pred = TRUE, extract = I)
folds <- vfold_cv(train_data_tbl)

cat_res <- finalize_workflow(cat_wflw, select_best(cat_tuned)) %>% 
  fit_resamples(folds, control = ctrl)

xgboost_res <- finalize_workflow(xgboost_wflw, select_best(xgboost_tuned)) %>% 
  fit_resamples(folds, control = ctrl)

lightgbm_res <- finalize_workflow(lightgbm_wflw, select_best(lightgbm_tuned)) %>% 
  fit_resamples(folds, control = ctrl)


# 2.4.2 Split conformal ----
catboost_conformal_split <- int_conformal_split(catboost_fin, val_data_tbl)
xgboost_conformal_split <- int_conformal_split(xgboost_fin, val_data_tbl)
lightgbm_conformal_split <- int_conformal_split(lightgbm_fin, val_data_tbl)

# 2.4.2 CV+ ----
catboost_cv_int <- int_conformal_cv(cat_res)
xgboost_cv_int  <- int_conformal_cv(xgboost_res)
lightgbm_cv_int <- int_conformal_cv(lightgbm_res)


coverage <- function(x) {
  x %>% 
    mutate(in_bound = .pred_lower <= dbwt & .pred_upper >= dbwt) %>% 
    summarise(coverage = mean(in_bound) * 100)
}




# 2.5.0 Coverage ----------------------------------------------------------

get_conformal_interval <- function(conf_split, test_data = test_data_tbl, level = 0.9) {
  
  predict(conf_split, test_data, level = level) %>% 
    bind_cols(test_data %>% select(dbwt))
}

catboost_test_split_tbl <- get_conformal_interval(catboost_conformal_split)
xgboost_test_split_tbl  <- get_conformal_interval(xgboost_conformal_split)
lightgbm_test_split_tbl  <- get_conformal_interval(lightgbm_conformal_split)

catboost_test_cv_tbl <- get_conformal_interval(catboost_cv_int)
xgboost_test_cv_tbl  <- get_conformal_interval(xgboost_cv_int)
lightgbm_test_cv_tbl  <- get_conformal_interval(lightgbm_cv_int)


# Coverage
coverage(catboost_test_split_tbl)
coverage(xgboost_test_split_tbl)
coverage(lightgbm_test_split_tbl)
coverage(catboost_test_cv_tbl)
coverage(xgboost_test_cv_tbl)
coverage(lightgbm_test_cv_tbl)


# XGBoost - Conformal CV+ 90.1 in coverate



# 2.6.0 Winkler -----------------------------------------------------------
alpha_winkler <- 0.1

xgboost_test_cv_tbl %>% 
  mutate(
    winkler_score = 
      case_when(
        dbwt >= .pred_lower & dbwt <= .pred_upper ~ .pred_upper - .pred_lower,
        dbwt < .pred_lower ~ (.pred_upper - .pred_lower) + 2/alpha_winkler * (.pred_lower - dbwt),
        TRUE ~ (.pred_upper - .pred_lower) + 2/alpha_winkler * (dbwt - .pred_upper)
      )
  )


# Refit before submitting
full_folds <- data_tbl %>% vfold_cv(strata = dbwt)

ctrl  <- control_resamples(save_pred = TRUE, extract = I, allow_par = TRUE)

modeltime::parallel_start(10)

xgboost_res_final <- finalize_workflow(xgboost_wflw, select_best(xgboost_tuned)) %>% 
  fit_resamples(full_folds, control = ctrl)

modeltime::parallel_stop()
gc()

xgboost_cv_final_int  <- int_conformal_cv(xgboost_res_final)


# 3.7.0 Final fit and prediction ------------------------------------------

final_pred_tbl <- predict(xgboost_cv_final_int, test_set_data_tbl, level = 0.9) %>% bind_cols(test_set_data_tbl %>% select(id))

n_submissions <- list.files("01_submission/", pattern = ".csv") %>% length()
n_submissions <- n_submissions + 1

file_path <- paste0("01_submission/", n_submissions, "_submission.csv")

final_pred_tbl %>% 
  select(id, .pred_lower, .pred_upper) %>% 
  set_names("id", "pi_lower", "pi_upper") %>% 
  write_csv(file_path)
