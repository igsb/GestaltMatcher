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

out_distances <- paste0(args[5],".RData")

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

###### calculate distances ########
distances <- mclapply(1:length(df$image_ids),function(i){sapply(df$image_ids[-i],function(id2){help_cosine_dist(df$image_ids[i],id2)})},mc.cores=50)

names(distances) = df$image_ids

###### save distances ########
save(distances,file=out_distances)