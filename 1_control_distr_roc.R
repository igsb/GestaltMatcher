library(reticulate)
library(data.table)
library(tidyverse)
library(lsa)
library(boot)
library(MASS)
library(plotly)
library(pROC)
library(parallel)


###### Input data #######
args <- commandArgs(T)

embeddings_file <- args[1]
syndromes_data_file <- args[2]
image_info_file <- args[3]
gallery_images_file <- args[4]
distances_file <- args[5]

out_roc <- paste0(args[6],".RData")

####### Read input data #########
set.seed(1)

pd <- import("pandas")

p_data <- as.data.frame(pd$read_pickle(embeddings_file))

names(p_data) <- paste(str_split(names(p_data),"\\.",simplify = T)[,1],rep(1:12,times=ncol(p_data)/12),sep=".")

data <- fread(syndromes_data_file)

data <- data %>% mutate(image_ids = str_remove_all(image_ids,"\\[")) %>% mutate(image_ids = str_remove_all(image_ids,"\\]")) %>% mutate(image_ids = str_remove_all(image_ids,"\\'")) %>% mutate(image_ids = str_remove_all(image_ids," ")) %>% mutate(image_ids = str_split(image_ids,",")) %>%
  mutate(subject_ids = str_remove_all(subject_ids,"\\[")) %>% mutate(subject_ids = str_remove_all(subject_ids,"\\]")) %>% mutate(subject_ids = str_remove_all(subject_ids,"\\'")) %>% mutate(subject_ids = str_remove_all(subject_ids," ")) %>% mutate(subject_ids = str_split(subject_ids,","))

image_info <- fread(image_info_file)

image_ids <- unique(str_split(names(p_data),"\\.",simplify = T)[,1])

####### define cosine functions ########
cosine_dist <- function(x,y){
  as.numeric(1-cosine(x,y))
}

help_cosine_dist <- function(id1, id2){
  out <- mean(sapply(1:12,function(i){cosine_dist(p_data[[paste(id1,i,sep=".")]],p_data[[paste(id2,i,sep=".")]])}))
  out
}

###### create data frame #######

df <- data.frame(id=str_remove_all(image_ids,"X"))
df$image_ids = image_ids
df$synd_id=NA
df$synd_name=NA
df$subj_id=NA

for(i in 1:nrow(df)){
  if(length(data$syndrome_id[sapply(data$image_ids,function(x){df$id[i] %in% x})])>1)print(i)
  if(sum(sapply(data$image_ids,function(x){df$id[i] %in% x}))==1){
    df$synd_id[i] = data$syndrome_id[sapply(data$image_ids,function(x){df$id[i] %in% x})]
    df$synd_name[i] = data$syndrome_name[sapply(data$image_ids,function(x){df$id[i] %in% x})]
    df$subj_id[i] = image_info$patient_id[image_info$image_id==df$id[i]]
    # df$synd_score[i] = data$score[sapply(data$image_ids,function(x){df$id[i] %in% x})]
  }else{
    next
  }
}

##### remove images without known syndrome #####

df <- df %>% filter(!is.na(synd_id))

###### filter for images not trained on ######
gallery_images <- fread(gallery_images_file)

df_all <- df

df <- df %>% filter(!id %in% gallery_images$image_id)

### extract syndromes + number of patients per syndrome + divide syndromes into 5 groups for cross validation
syndromes <- data.frame(synd_name=unique(df$synd_name))

syndromes$subjects = NA
for(i in 1:nrow(syndromes)){
  syndromes$subjects[i] = length(unique(df$subj_id[df$synd_name==syndromes$synd_name[i]]))
}

syndromes <- syndromes %>% filter(subjects >1 )

cv_help <- runif(nrow(syndromes))
syndromes <- syndromes %>% mutate(cv_group = case_when(cv_help < 0.2 ~ 1,
                                                       cv_help >=0.2 & cv_help < 0.4 ~ 2,
                                                     cv_help >=0.4 & cv_help < 0.6 ~ 3,
                                                       cv_help >=0.6 & cv_help < 0.8 ~ 4,
                                                       cv_help >=0.8~ 5
))

### load distances ######

load(distances_file)

### define functions for roc analysis (training and evaluation on test set) for 5fold CV

train_roc <- function(i){
  ### restrict syndromes to all but cv_group i
  syndromes_tmp <- syndromes %>% filter(cv_group != i)
  ### create control distribution for same syndromes
  control_same_synd <- data.frame(synd= NA, sample_size1=NA, sample_size2=NA, mean_pw_dist=NA)
  nsimu_per_synd <- 10
  m=1
  for(s in 1:nrow(syndromes_tmp)){
    # sample nsimu_per_synd splits per syndrome
    stop=min(nsimu_per_synd,choose(syndromes_tmp$subjects[s],round(syndromes_tmp$subjects[s]/2))/2)
    for(i in 1:stop){
      control_same_synd[m,1]=syndromes_tmp$synd_name[s]
      sample <- data.frame(id=df$image_ids[df$synd_name==syndromes_tmp$synd_name[s]],group=rbinom(sum(df$synd_name==syndromes_tmp$synd_name[s]),1,0.5))
      while(sum(sample$group==0)==0|sum(sample$group==1)==0|!is_empty(intersect(sample$id[sample$group==0],sample$id[sample$group==1]))){sample <- data.frame(id=df$image_ids[df$synd_name==syndromes_tmp$synd_name[s]],group=rbinom(sum(df$synd_name==syndromes_tmp$synd_name[s]),1,0.5))}
      control_same_synd[m,2:3]=c(sum(sample$group==0),sum(sample$group==1))
      
      control_same_synd$mean_pw_dist[m] <-mean(unlist(lapply(sample$id[sample$group==0],function(i){distances[[i]][sample$id[sample$group==1]]})))
      m = m+1
    }
  }
  
  ########## construct control distribution for different syndrome cluster ############
  control_diff_synd <- data.frame(synd1= NA, synd2 = NA, sample_size1=NA, sample_size2=NA, mean_pw_dist=NA)
  nsimu_per_synd <- 5
  m=1
  for(s in 1:(nrow(syndromes_tmp)-1)){
    for(t in (s+1):nrow(syndromes_tmp)){
      # sample nsimu_per_synd splits per syndrome combination
      stop = min(nsimu_per_synd,sum(df$synd_name==syndromes_tmp$synd_name[s])*sum(df$synd_name==syndromes_tmp$synd_name[t]))
      for(i in 1:stop){
        control_diff_synd[m,1:2]=syndromes_tmp$synd_name[c(s,t)]
        sample1 <- df$image_ids[df$synd_name==syndromes_tmp$synd_name[s]][rbinom(sum(df$synd_name==syndromes_tmp$synd_name[s]),1,0.5)==1]
        while(is_empty(sample1)){ sample1 <-  df$image_ids[df$synd_name==syndromes_tmp$synd_name[s]][rbinom(sum(df$synd_name==syndromes_tmp$synd_name[s]),1,0.5)==1]}
        sample2 <-  df$image_ids[df$synd_name==syndromes_tmp$synd_name[t]][rbinom(sum(df$synd_name==syndromes_tmp$synd_name[t]),1,0.5)==1]
        while(is_empty(sample2)){ sample2 <- df$image_ids[df$synd_name==syndromes_tmp$synd_name[t]][rbinom(sum(df$synd_name==syndromes_tmp$synd_name[t]),1,0.5)==1]}
        control_diff_synd[m,3:4]=c(length(sample1),length(sample2))
        
        control_diff_synd$mean_pw_dist[m] <-mean(unlist(lapply(sample1,function(i){distances[[i]][sample2]})))
        m = m+1
      }
    }
  }
  
  distributions <- data.frame(mean_pw_dist = c(unique(control_same_synd$mean_pw_dist),unique(control_diff_synd$mean_pw_dist)),
                              distribution = c(rep("same syndromes",length(unique(control_same_synd$mean_pw_dist))),
                                               rep("different syndromes",length(unique(control_diff_synd$mean_pw_dist)))))
  
  #### roc analysis
  
  distr <- distributions
  
  distr <- distr %>% mutate(truth = case_when(distribution=="same syndromes" ~ 0,
                                              distribution=="different syndromes" ~ 1))
  
  roc_distr <- roc(distr$truth,distr$mean_pw_dist, plot=T, auc=T, print.auc=T)
  
  ## use Youden-Index for decision: max(sensitivity+specificity-1)
  
  opt_thresh<- roc_distr$thresholds[which.max(roc_distr$sensitivities+roc_distr$specificities-1)]
  
  sensitivity_opt_thresh <- roc_distr$sensitivities[which.max(roc_distr$sensitivities+roc_distr$specificities-1)]
  
  specificity_opt_thresh <- roc_distr$specificities[which.max(roc_distr$sensitivities+roc_distr$specificities-1)]
  
  out <- list(roc=roc_distr,distributions=distributions,opt_thresh=opt_thresh,Youden=sensitivity_opt_thresh+specificity_opt_thresh-1)
  out
}

test_roc <- function(i,c){
  ### restrict syndromes to all but cv_group i
  syndromes_tmp <- syndromes %>% filter(cv_group == i)
  ### create control distribution for same syndromes
  control_same_synd <- data.frame(synd= NA, sample_size1=NA, sample_size2=NA, mean_pw_dist=NA)
  nsimu_per_synd <- 10
  m=1
  for(s in 1:nrow(syndromes_tmp)){
    # sample nsimu_per_synd splits per syndrome
    stop=min(nsimu_per_synd,choose(syndromes_tmp$subjects[s],round(syndromes_tmp$subjects[s]/2))/2)
    for(i in 1:stop){
      control_same_synd[m,1]=syndromes_tmp$synd_name[s]
      sample <- data.frame(id=df$image_ids[df$synd_name==syndromes_tmp$synd_name[s]],group=rbinom(sum(df$synd_name==syndromes_tmp$synd_name[s]),1,0.5))
      while(sum(sample$group==0)==0|sum(sample$group==1)==0|!is_empty(intersect(sample$id[sample$group==0],sample$id[sample$group==1]))){sample <- data.frame(id=df$image_ids[df$synd_name==syndromes_tmp$synd_name[s]],group=rbinom(sum(df$synd_name==syndromes_tmp$synd_name[s]),1,0.5))}
      control_same_synd[m,2:3]=c(sum(sample$group==0),sum(sample$group==1))
      
      control_same_synd$mean_pw_dist[m] <-mean(unlist(lapply(sample$id[sample$group==0],function(i){distances[[i]][sample$id[sample$group==1]]})))
      m = m+1
    }
  }
  
  ########## construct control distribution for different syndrome cluster ############
  control_diff_synd <- data.frame(synd1= NA, synd2 = NA, sample_size1=NA, sample_size2=NA, mean_pw_dist=NA)
  nsimu_per_synd <- 5
  m=1
  for(s in 1:(nrow(syndromes_tmp)-1)){
    for(t in (s+1):nrow(syndromes_tmp)){
      # sample nsimu_per_synd splits per syndrome combination
      stop = min(nsimu_per_synd,sum(df$synd_name==syndromes_tmp$synd_name[s])*sum(df$synd_name==syndromes_tmp$synd_name[t]))
      for(i in 1:stop){
        control_diff_synd[m,1:2]=syndromes_tmp$synd_name[c(s,t)]
        sample1 <- df$image_ids[df$synd_name==syndromes_tmp$synd_name[s]][rbinom(sum(df$synd_name==syndromes_tmp$synd_name[s]),1,0.5)==1]
        while(is_empty(sample1)){ sample1 <-  df$image_ids[df$synd_name==syndromes_tmp$synd_name[s]][rbinom(sum(df$synd_name==syndromes_tmp$synd_name[s]),1,0.5)==1]}
        sample2 <-  df$image_ids[df$synd_name==syndromes_tmp$synd_name[t]][rbinom(sum(df$synd_name==syndromes_tmp$synd_name[t]),1,0.5)==1]
        while(is_empty(sample2)){ sample2 <- df$image_ids[df$synd_name==syndromes_tmp$synd_name[t]][rbinom(sum(df$synd_name==syndromes_tmp$synd_name[t]),1,0.5)==1]}
        control_diff_synd[m,3:4]=c(length(sample1),length(sample2))
        
        control_diff_synd$mean_pw_dist[m] <-mean(unlist(lapply(sample1,function(i){distances[[i]][sample2]})))
        m = m+1
      }
    }
  }
  
  specificity = sum(unique(control_same_synd$mean_pw_dist)<c)/length(unique(control_same_synd$mean_pw_dist))
  sensitivity = sum(unique(control_diff_synd$mean_pw_dist)>=c)/length(unique(control_diff_synd$mean_pw_dist))
  
  out <- list(sensitivity = sensitivity, specificity = specificity, Youden = sensitivity + specificity -1, 
              control_diff_synd = control_diff_synd[!duplicated(control_diff_synd$mean_pw_dist),], 
              control_same_synd=control_same_synd[!duplicated(control_same_synd$mean_pw_dist),])
  out
}

##### run cross validation

rocs <- list()
eval <- list()

#for(cv in 1:5){
#  rocs[[cv]] <- train_roc(cv)
#  eval[[cv]] <- test_roc(cv,rocs[[cv]][["opt_thresh"]])
#}

rocs <- parallel::mclapply(1:5,train_roc,mc.cores = 50, mc.set.seed = FALSE)

test_roc_cv <- function(cv){test_roc(cv,rocs[[cv]][['opt_thresh']])}

eval <- mclapply(1:5,test_roc_cv,mc.cores = 50, mc.set.seed = FALSE)

i_opt <- which.max(c(eval[[1]][["Youden"]],eval[[2]][["Youden"]],eval[[3]][["Youden"]],eval[[4]][["Youden"]],eval[[5]][["Youden"]]))

distributions <- rocs[[i_opt]]$distributions

#### roc analysis

roc_distr <- rocs[[i_opt]]$roc

## use Youden-Index for decision: max(sensitivity+specificity-1)

opt_thresh<- rocs[[i_opt]]$opt_thresh

sensitivity_opt_thresh <- roc_distr$sensitivities[which.max(roc_distr$sensitivities+roc_distr$specificities-1)]

specificity_opt_thresh <- roc_distr$specificities[which.max(roc_distr$sensitivities+roc_distr$specificities-1)]

save(distributions,roc_distr,opt_thresh,sensitivity_opt_thresh,specificity_opt_thresh, i_opt, df, syndromes, rocs, eval, file=out_roc)
