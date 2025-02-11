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

roc_file <- args[1]
distances_file <- args[2]

out_split_lump <- args[3]

####### Read input data #########
set.seed(1)
load(roc_file)
load(distances_file)

syndromes_test <- syndromes %>% filter(cv_group == i_opt)

#### splitting

syndromes_comb <- combn(syndromes_test$synd_name,2)

syndromes_comb <- data.frame(synd1 = syndromes_comb[1,], synd2 = syndromes_comb[2,])

syndromes_comb$mean_pw_dist = numeric(nrow(syndromes_comb))

syndromes_comb$percentage = numeric(nrow(syndromes_comb))

for(s in 1:nrow(syndromes_comb)){
  syndromes_comb$mean_pw_dist[s] <-   mean(unlist(lapply(df$image_ids[df$synd_name==syndromes_comb$synd1[s]],function(i){distances[[i]][df$image_ids[df$synd_name==syndromes_comb$synd2[s]]]})))

  n_samples <- 100
  mean_pw_dist_tmp <- numeric(n_samples) 
  #sample_sizes <- data.frame(n_mctt=numeric(n_samples), n_mntt=numeric(n_samples))
  for(i in 1:n_samples){
    n1 <- round(runif(1,min=2,max=length(df$image_ids[df$synd_name==syndromes_comb$synd1[s]])))
    #sample_sizes$n_mctt[i]=n_mctt
    sample1 <- sample(df$image_ids[df$synd_name==syndromes_comb$synd1[s]],n1)
    n2 <- round(runif(1,min=2,max=length(df$image_ids[df$synd_name==syndromes_comb$synd2[s]])))
    # sample_sizes$n_mntt[i]=n_mntt
    sample2 <- sample(df$image_ids[df$synd_name==syndromes_comb$synd2[s]],n2)
    mean_pw_dist_tmp[i] <- mean(unlist(lapply(sample1,function(i){distances[[i]][sample2]})))
  }
  syndromes_comb$percentage[s] = sum(unique(mean_pw_dist_tmp)> opt_thresh)/length(unique(mean_pw_dist_tmp))
}

### lumping

syndromes_lump <- data.frame(synd1 = syndromes_test$synd_name, subj = syndromes_test$subjects)

syndromes_lump$mean_pw_dist = numeric(nrow(syndromes_lump))

syndromes_lump$percentage = numeric(nrow(syndromes_lump))

for(s in 1:nrow(syndromes_lump)){
  sample <- data.frame(id=df$image_ids[df$synd_name==syndromes_lump$synd1[s]],group=rbinom(sum(df$synd_name==syndromes_lump$synd1[s]),1,0.5))
  while(sum(sample$group==0)==0|sum(sample$group==1)==0|!is_empty(intersect(sample$id[sample$group==0],sample$id[sample$group==1]))){
    sample <- data.frame(id=df$image_ids[df$synd_name==syndromes_lump$synd1[s]],group=rbinom(sum(df$synd_name==syndromes_lump$synd1[s]),1,0.5))
    }
  
  syndromes_lump$mean_pw_dist[s] <- mean(unlist(lapply(sample$id[sample$group==0],function(i){distances[[i]][sample$id[sample$group==1]]})))
  
  n_samples <- 100
  mean_pw_dist_tmp <- numeric(n_samples) 
  #sample_sizes <- data.frame(n_mctt=numeric(n_samples), n_mntt=numeric(n_samples))
  for(i in 1:n_samples){
    n1 <- round(runif(1,min=1,max=sum(sample$group==0)))
    #sample_sizes$n_mctt[i]=n_mctt
    sample1 <- sample(sample$id[sample$group==0],n1)
    n2 <- round(runif(1,min=1,max=sum(sample$group==1)))
    # sample_sizes$n_mntt[i]=n_mntt
    sample2 <- sample(sample$id[sample$group==1],n2)
    mean_pw_dist_tmp[i] <- mean(unlist(lapply(sample1,function(i){distances[[i]][sample2]})))
  }
  syndromes_lump$percentage[s] = sum(unique(mean_pw_dist_tmp)> opt_thresh)/length(unique(mean_pw_dist_tmp))
}

syndromes_all <- full_join(syndromes_comb,syndromes_lump)
syndromes_all$group=case_when(is.na(syndromes_all$synd2)~"same syndromes",!is.na(syndromes_all$synd2)~"different syndromes")

### safe syndromes_all
save(syndromes_all,file=paste0(out_split_lump,".RData"))

