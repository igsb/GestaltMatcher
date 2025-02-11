library(reticulate)
library(data.table)
library(tidyverse)
library(lsa)
library(boot)
library(MASS)
library(plotly)
library(pROC)


###### Input data #######
args <- commandArgs(T)

embeddings_file <- args[1]
syndromes_data_file <- args[2]
image_info_file <- args[3]
gallery_images_file <- args[4]
distances_file <- args[5]
cohort_embeddings_file <- args[6]
cohort_name <- args[7]
out <- args[8]

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

### create df_cohort

p_cohort <- as.data.frame(pd$read_pickle(cohort_embeddings_file))

names_tmp <- str_split(names(p_cohort),"\\.",simplify = T)[,1:2]

names_tmp <- paste(names_tmp[,1],names_tmp[,2],sep="_")

names_tmp <- str_remove_all(names_tmp,"_c")

names(p_cohort) <- paste(names_tmp,rep(1:12,times=ncol(p_cohort)/12),sep=".")

image_ids_cohort <- unique(str_split(names(p_cohort),"\\.",simplify = T)[,1])

#### distance functions
cosine_dist <- function(x,y){
  as.numeric(1-cosine(x,y))
}

help_cosine_dist <- function(id1, id2){
  out <- mean(sapply(1:12,function(i){cosine_dist(p_data[[paste(id1,i,sep=".")]],p_data[[paste(id2,i,sep=".")]])}))
  out
}

help_cosine_dist_cohort <- function(id1, id2){
  out <- mean(sapply(1:12,function(i){cosine_dist(p_cohort[[paste(id1,i,sep=".")]],p_cohort[[paste(id2,i,sep=".")]])}))
  out
}

#### load distances
load(distances_file)

####
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

### filter for individuals GestaltMatcher was not trained on
gallery_images <- fread(gallery_images_file)

df_all <- df

df <- df %>% filter(!id %in% gallery_images$image_id)

df <- df %>% filter(!is.na(synd_id))

syndromes <- data.frame(synd_name=unique(df$synd_name))

syndromes$subjects = NA
for(i in 1:nrow(syndromes)){
  syndromes$subjects[i] = length(unique(df$subj_id[df$synd_name==syndromes$synd_name[i]]))
}

syndromes <- syndromes %>% filter(subjects >1)

df <- df %>% filter(synd_name %in% syndromes$synd_name)

#### sample distribution of random controls

control_random <- data.frame(sample_size1=rep(10, 10000), mean_pw_dist=NA)

for(m in 1:nrow(control_random)){
  sample1 <- sample(df$image_ids,control_random$sample_size1[m])
  mean_pw_dist_tmp <- mean(unlist(lapply(1:(length(sample1)-1),function(i){distances[[sample1[i]]][sample1[(i+1):length(sample1)]]})))
  while(mean_pw_dist_tmp %in% control_random$mean_pw_dist){
    sample1 <- sample(df$image_ids,control_random$sample_size1[m])
    mean_pw_dist_tmp <- mean(unlist(lapply(1:(length(sample1)-1),function(i){distances[[sample1[i]]][sample1[(i+1):length(sample1)]]})))
  }
  control_random$mean_pw_dist[m] <- mean_pw_dist_tmp
}

### calculate distances within cohort

distances_cohort <- lapply(1:length(image_ids_cohort),function(i){sapply(image_ids_cohort[-i],function(id2){help_cosine_dist_cohort(image_ids_cohort[i],id2)})})

names(distances_cohort)=image_ids_cohort

######################### cohort analysis ########################################

cohort_mean_pw_dist_all_ids <- mean(unlist(lapply(1:(length(image_ids_cohort)-1),function(i){distances_cohort[[image_ids_cohort[i]]][image_ids_cohort[(i+1):length(image_ids_cohort)]]})),na.rm=T)

cuts = data.frame(value = c("Threshold", cohort_name), 
                  mean_pw_distance = c(quantile(control_random$mean_pw_dist, probs=0.05),cohort_mean_pw_dist_all_ids))

color_palette <- c(RColorBrewer::brewer.pal(3,"Set1")[1:2],RColorBrewer::brewer.pal(9,"YlOrBr")[5])

ggplot(control_random, aes(x=mean_pw_dist, fill = "blue", col = "blue")) +
  geom_density(alpha=0.7) +
  scale_fill_manual(values = color_palette[c(3,1)])+
  scale_color_manual(values = color_palette[c(3,1)])+
  geom_vline(data = cuts %>% filter(value %in% c("Threshold",cohort_name)), aes(xintercept=mean_pw_distance),col=c("black",color_palette[c(3)]),show.legend = F,linewidth = 1.2)+
  theme_minimal()+
  theme(text=element_text(size=40),axis.text = element_text(size=24), legend.position = "none")+
  scale_x_continuous(breaks = seq(0.3,1.1,0.1))+
  labs(x="Mean pairwise distance", y="density")

ggsave(paste0(out,".svg"),width=24, height=16, dpi=300)

quant <- sum(control_random$mean_pw_dist > cohort_mean_pw_dist_all_ids)/nrow(control_random)

quant

save.image(paste0(out,".RData"))

