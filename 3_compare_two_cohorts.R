library(reticulate)
library(data.table)
library(tidyverse)
library(lsa)
library(boot)
library(MASS)
library(plotly)
library(pROC)
library(parallel)
library(RColorBrewer)
library(pals)

###### Input data #######
args <- commandArgs(T)

cohort1_embeddings_file <- args[1]
cohort2_embeddings_file <- args[2]
gene <- args[3]
cohort1_name <- args[4]
cohort2_name <- args[5]
roc_file <- args[6]
split_lump_file <- args[7]
out <- args[8]

####### Read input data #########
set.seed(1)

pd <- import("pandas")

cohort1 <- as.data.frame(pd$read_pickle(cohort1_embeddings_file))

names_tmp <- str_split(names(cohort1),"\\.",simplify = T)[,1:2]

names_tmp <- paste(names_tmp[,1],names_tmp[,2],sep="_")

names_tmp <- str_remove_all(names_tmp,"_c")

names(cohort1) <- paste(names_tmp,rep(1:12,times=ncol(cohort1)/12),sep=".")

image_ids_cohort1 <- unique(str_split(names(cohort1),"\\.",simplify = T)[,1])

cohort2 <- as.data.frame(pd$read_pickle(cohort2_embeddings_file))

names_tmp <- str_split(names(cohort2),"\\.",simplify = T)[,1:2]

names_tmp <- paste(names_tmp[,1],names_tmp[,2],sep="_")

names_tmp <- str_remove_all(names_tmp,"_c")

names(cohort2) <- paste(names_tmp,rep(1:12,times=ncol(cohort2)/12),sep=".")

image_ids_cohort2 <- unique(str_split(names(cohort2),"\\.",simplify = T)[,1])

## for MN1, exclude images 1260c09_2 and M1273C15

image_ids_cohort1 <- image_ids_cohort1[!image_ids_cohort1 %in% c("M1273C15", "X1260C09_2")]
image_ids_cohort2 <- image_ids_cohort2[!image_ids_cohort2 %in% c("M1273C15", "X1260C09_2")]

####### define cosine functions ########
cosine_dist <- function(x,y){
  as.numeric(1-cosine(x,y))
}

help_cosine_dist_cohort1_cohort2 <- function(id1, id2){
  out <- mean(sapply(1:12,function(i){cosine_dist(cohort1[[paste(id1,i,sep=".")]],cohort2[[paste(id2,i,sep=".")]])}))
  out
}

help_cosine_dist_cohort1 <- function(id1, id2){
  out <- mean(sapply(1:12,function(i){cosine_dist(cohort1[[paste(id1,i,sep=".")]],cohort1[[paste(id2,i,sep=".")]])}))
  out
}

help_cosine_dist_cohort2 <- function(id1, id2){
  out <- mean(sapply(1:12,function(i){cosine_dist(cohort2[[paste(id1,i,sep=".")]],cohort2[[paste(id2,i,sep=".")]])}))
  out
}

###### load roc results
load(roc_file)

distributions <- distributions %>% mutate(distribution = str_replace_all(distribution,"same","Same")) %>% 
  mutate(distribution = str_replace_all(distribution,"different","Different"))
  
####### calculate distances

distances_cohort1_cohort2 <- lapply(image_ids_cohort1,function(i){sapply(image_ids_cohort2,function(id2){help_cosine_dist_cohort1_cohort2(i,id2)})})

names(distances_cohort1_cohort2) = image_ids_cohort1

pw_dist_cohort1_cohort2 <- unique(unlist(distances_cohort1_cohort2))

pw_dist_cohort1_cohort2 <- pw_dist_cohort1_cohort2[pw_dist_cohort1_cohort2!=0]

distances_cohort1 <- lapply(image_ids_cohort1,function(i){sapply(image_ids_cohort1,function(id2){help_cosine_dist_cohort1(i,id2)})})

names(distances_cohort1) = image_ids_cohort1

pw_dist_cohort1 <- unique(unlist(distances_cohort1))

pw_dist_cohort1 <- pw_dist_cohort1[pw_dist_cohort1!=0]

distances_cohort2 <- lapply(image_ids_cohort2,function(i){sapply(image_ids_cohort2,function(id2){help_cosine_dist_cohort2(i,id2)})})

names(distances_cohort2) = image_ids_cohort2

pw_dist_cohort2 <- unique(unlist(distances_cohort2))

pw_dist_cohort2 <- pw_dist_cohort2[pw_dist_cohort2!=0]

df_pw_dist <- data.frame(dist=unique(unlist(pw_dist_cohort1)),cohort=cohort1_name) %>%
  add_row(dist=unique(unlist(pw_dist_cohort2)),cohort=cohort2_name) %>%
  add_row(dist=unique(unlist(pw_dist_cohort1_cohort2)),cohort=paste0(cohort1_name," vs. ",cohort2_name))
df_pw_dist <- df_pw_dist %>% filter(!is.na(dist))

ggplot(df_pw_dist,aes(x=dist,fill=cohort,col=cohort))+
  geom_histogram(alpha=0.7,position = "dodge")+
  geom_vline(xintercept = opt_thresh)+
  theme_minimal()+
  theme(text=element_text(size=40))

ggsave(paste0(out,"_hist.svg"),width=24, height=16, dpi=300)
#ggsave(paste0(out,"_hist.png"),width=24, height=16, dpi=300)

####### subsampling for threshold

n_samples <- 100
mean_pw_dist_cohort1_cohort2 <- numeric(n_samples) 
#sample_sizes <- data.frame(n_cohort1=numeric(n_samples), n_cohort2=numeric(n_samples))
for(i in 1:n_samples){
  n_cohort1 <- round(runif(1,min=2,max=length(image_ids_cohort1)))
  #sample_sizes$n_cohort1[i]=n_cohort1
  sample1 <- sample(image_ids_cohort1,n_cohort1)
  n_cohort2 <- round(runif(1,min=2,max=length(image_ids_cohort2)))
  # sample_sizes$n_cohort2[i]=n_cohort2
  sample2 <- sample(image_ids_cohort2,n_cohort2)
  mean_pw_dist_cohort1_cohort2[i] <- mean(unlist(lapply(sample1,function(i){distances_cohort1_cohort2[[i]][sample2]})))
}

mean_pw_dist_cohort1_cohort2_complete <-   mean(unlist(lapply(image_ids_cohort1,function(i){distances_cohort1_cohort2[[i]][image_ids_cohort2]})))
distributions = distributions %>% add_row(data.frame(mean_pw_dist=mean_pw_dist_cohort1_cohort2, distribution =paste0(cohort1_name,"/",cohort2_name)))
cuts = data.frame(value = c("threshold",paste0(cohort1_name,"/",cohort2_name)), mean_pw_distance = c(opt_thresh,mean_pw_dist_cohort1_cohort2_complete))

color_palette <- c(RColorBrewer::brewer.pal(3,"Set1")[1:2],RColorBrewer::brewer.pal(9,"YlOrBr")[5])

ggplot(distributions, aes(x=mean_pw_dist, fill = distribution, col=distribution)) +
  geom_density(alpha=0.7) +
  scale_fill_manual(values = color_palette[c(3,2,1)],breaks=c(paste0(cohort1_name,"/",cohort2_name),"Same syndromes","Different syndromes"))+
  scale_color_manual(values = color_palette[c(3,2,1)],breaks=c(paste0(cohort1_name,"/",cohort2_name),"Same syndromes","Different syndromes"))+
  geom_vline(data = cuts, aes(xintercept=mean_pw_distance),col=c("black",color_palette[3]),show.legend = F,linewidth = 1.2)+
  theme_minimal()+
  theme(text=element_text(size=40),axis.text = element_text(size=24), legend.position = "bottom", legend.title = element_blank())+
 # annotate("text",x=mean_pw_dist_cohort1_cohort2_complete+0.11,y=round(max(density(distributions$mean_pw_dist[distributions$distribution==paste0(cohort1_name,"/",cohort2_name)])$y)/5)*5,label=paste0(cohort1_name,"/",cohort2_name),color=color_palette[3],size=16)+
  annotate("text",x=opt_thresh-0.08,y=round(max(density(distributions$mean_pw_dist[distributions$distribution==paste0(cohort1_name,"/",cohort2_name)])$y)/5)*5,label="Threshold",color="black",size=16)+
  scale_x_continuous(breaks=seq(round(min(distributions$mean_pw_dist),digits=1)-0.1,round(max(distributions$mean_pw_dist),digits=1)+0.1,0.1))+
  labs(x="Mean pairwise distance", y="Density")


ggsave(paste0(out,".svg"),width=24, height=16, dpi=300)


#### percentage above threshold
perc_above_thresh <- sum(unique(mean_pw_dist_cohort1_cohort2)>opt_thresh)/length(unique(mean_pw_dist_cohort1_cohort2))

### create  boxplots

n_samples <- 100
sample_sizes <- 1:min(length(image_ids_cohort1),length(image_ids_cohort2))
results_per_sample_size <- data.frame(sample_size=NA, mean_pw_dist_cohort1_cohort2 = NA)
for(size in sample_sizes){
  results_tmp <- data.frame(sample_size = rep(sample_sizes[size],n_samples), mean_pw_dist_cohort1_cohort2 = NA)
  for(i in 1:n_samples){
    sample1 <- sample(image_ids_cohort1,size)
    sample2 <- sample(image_ids_cohort2,size)
    results_tmp$mean_pw_dist_cohort1_cohort2[i] <- mean(unlist(lapply(sample1,function(i){distances_cohort1_cohort2[[i]][sample2]})))
  }
  results_per_sample_size <- results_per_sample_size %>% add_row(results_tmp)
}
results_per_sample_size <- results_per_sample_size[-1,]

ggplot(results_per_sample_size,aes(x=sample_size, y=mean_pw_dist_cohort1_cohort2, group = sample_size, fill= as.factor(sample_size)))+
  geom_boxplot()+
  theme_minimal()+
  theme(text=element_text(size=40),axis.text = element_text(size=24), legend.position = "none")+
  scale_x_continuous(breaks=sample_sizes)+
  scale_fill_manual(values = brewer.oranges(length(sample_sizes)*2)[-c((1:length(sample_sizes)/2),((2*length(sample_sizes)-length(sample_sizes)/2):2*length(sample_sizes)))])+
  geom_hline(yintercept = opt_thresh, col="black",linetype="dashed",linewidth=1.5)+
  annotate("text",x=max(sample_sizes)-0.5,y=opt_thresh-0.02,label="Threshold",color="black",size=16)+
  labs(x="Sample size", y="Mean pairwise distance")

ggsave(paste0(out,"_boxplots.svg"),width=24, height=16, dpi=300)
#ggsave(paste0(out,"_boxplots.png"),width=24, height=16, dpi=300)


#### put it into splitting and lumping plot
load(split_lump_file)

syndromes_all <- syndromes_all %>% mutate(group = str_replace_all(group,"same","Same")) %>% mutate(group = str_replace_all(group,"different","Different"))

ggplot(syndromes_all, aes(x=mean_pw_dist, y= percentage,col=group,shape=group))+
  scale_shape_discrete(breaks= c("Same syndromes", "Different syndromes"))+
  scale_color_manual(values=color_palette[c(2,1)],breaks= c("Same syndromes", "Different syndromes"))+
  geom_point(size=5,alpha=0.85)+
  geom_vline(xintercept = opt_thresh,col="red",linewidth=1.5)+
  geom_hline(yintercept = 0.5, col="red", linetype="dashed",linewidth=1.5)+
  geom_point(aes(x=mean_pw_dist_cohort1_cohort2_complete,y=perc_above_thresh), size=8, col= color_palette[3], fill= color_palette[3], alpha=0.85, shape=18,stat="unique")+
  annotate("text",x=mean_pw_dist_cohort1_cohort2_complete-0.04,y=perc_above_thresh-0.03,label=paste0(cohort1_name,"/",cohort2_name), size=14, col= color_palette[3])+
  labs(x="Mean pairwise distance", y="Percentage above threshold")+
  scale_x_continuous(breaks=seq(round(min(syndromes_all$mean_pw_dist),digits=1)-0.1,round(max(syndromes_all$mean_pw_dist),digits=1)+0.1,0.1))+
  theme_minimal()+
  theme(text=element_text(size=40),axis.text = element_text(size=40),legend.position = "bottom",legend.title = element_blank())

ggsave(paste0(out,"split_lump.svg"),width=24, height=16, dpi=300)
#ggsave(paste0(out,"split_lump.png"),width=24, height=16, dpi=300)

## calculate positive predictive value

validation_distributions <- data.frame(distribution = "same",
                                       mean_pw_dist = c(eval[[1]]$control_same_synd$mean_pw_dist,
                                                        eval[[2]]$control_same_synd$mean_pw_dist,
                                                        eval[[3]]$control_same_synd$mean_pw_dist,
                                                        eval[[4]]$control_same_synd$mean_pw_dist,
                                                        eval[[5]]$control_same_synd$mean_pw_dist)) %>%
  add_row(distribution = "different",
          mean_pw_dist = c(eval[[1]]$control_diff_synd$mean_pw_dist,
                           eval[[2]]$control_diff_synd$mean_pw_dist,
                           eval[[3]]$control_diff_synd$mean_pw_dist,
                           eval[[4]]$control_diff_synd$mean_pw_dist,
                           eval[[5]]$control_diff_synd$mean_pw_dist))


validation_distributions$d_in_interval = ifelse(validation_distributions$mean_pw_dist >= min(mean_pw_dist_cohort1_cohort2) & validation_distributions$mean_pw_dist <= max(mean_pw_dist_cohort1_cohort2), "d in interval", "d not in interval")

t_d_in_interval <- table(validation_distributions$d_in_interval, validation_distributions$distribution)

sensitivity_d_in_interval <- t_d_in_interval[1,1]/(t_d_in_interval[1,1]+t_d_in_interval[2,1])
specificity_d_in_interval <- t_d_in_interval[2,2]/(t_d_in_interval[1,2]+t_d_in_interval[2,2])

ppv_d_in_interval <- sensitivity_d_in_interval*0.5/(sensitivity_d_in_interval*0.5+(1-specificity_d_in_interval)*0.5)

results_table <- data.frame(name=c("mean pairwise distance", "percentage above threshold","threshold","sensitivity", "specificity", "ppv_d_in_interval"),
                            value=c(mean_pw_dist_cohort1_cohort2_complete, perc_above_thresh, opt_thresh, sensitivity_opt_thresh, specificity_opt_thresh, ppv_d_in_interval))
fwrite(results_table,file=paste0(out,"_results_table.txt"),sep="\t")

save(gene, cohort1_name, cohort2_name, mean_pw_dist_cohort1_cohort2_complete, mean_pw_dist_cohort1_cohort2, perc_above_thresh, file=paste0(out,".RData"))
