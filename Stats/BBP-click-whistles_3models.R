########################################################################
#    STATISTICS
#    Author : Loic LEHNHOFF
#    Adapted from Yannick OUTREMAN 
#    Agrocampus Ouest - 2020
#######################################################################
library(pscl)
library(MASS)
library(lmtest)
library(multcomp)
library(emmeans)
library(dplyr)        # "%>%" function
library(forcats)      # "fct_relevel" function
library(stringr)      # "gsub" function 
library(rcompanion)   # "fullPTable" function
library(multcompView) # "multcompLetters" function
library(ggplot2)
library(pgirmess)
#library(tidyquant)    # geom_ma() if rolling average needed


################# DATASET IMPORTS #####################################
folder <- './../'
whistles.dta <-read.table(file=paste0(folder, 
              'Whistles/Evaluation/whistles_durations.csv'),
              sep = ',', header=TRUE)
whistles.dta <- whistles.dta[order(whistles.dta$audio_names),]
bbp.dta <-read.table(file=paste0(folder, 
              'BBPs/Results/16-06-22_14h00_number_of_BBP.csv'),
              sep = ',', header=TRUE)
bbp.dta <- bbp.dta[order(bbp.dta$audio_names),]
clicks.dta <-read.table(file=paste0(folder,
              'Clicks/Results/projection_updated_number_of_clicks_02052022.csv'), #number_of_clicks_02052022.csv
              sep = ',', header=TRUE)
clicks.dta <- clicks.dta[order(clicks.dta$audio_names),]

# Merge files into 1 dataset
acoustic.dta <- clicks.dta
acoustic.dta$number_of_bbp <- bbp.dta$number_of_BBP
acoustic.dta$total_whistles_duration <- whistles.dta$total_whistles_duration
rm(whistles.dta, bbp.dta, clicks.dta)

# suppress "T" acoustic data (other groups not tested on our variables)
acoustic.dta <- acoustic.dta[acoustic.dta$acoustic!="T",]
# shuffle dataframe
acoustic.dta <- acoustic.dta[sample(1:nrow(acoustic.dta)), ]
acoustic.dta$acoustic <- factor(acoustic.dta$acoustic)

#################### DATA INSPECTION  #################################
# Data description
names(acoustic.dta)
# self explenatory except acoustic : correspond to the activation sequence.

# Look for obvious correlations
plot(acoustic.dta) # nothing that we can see

# Look for zero-inflation
100*sum(acoustic.dta$number_of_clicks == 0)/nrow(acoustic.dta)
100*sum(acoustic.dta$number_of_bbp == 0)/nrow(acoustic.dta)
100*sum(acoustic.dta$total_whistles_duration == 0)/nrow(acoustic.dta)
# 3.6%, 53.7% & 24.7% of our data are zeros. Will have to be dealt with.

# QUESTION: This study is aimed at understanding if dolphin's acoustic activity
# is influenced bytheir behavior, the emission of a pinger or a fishing net.

# Dependent variables (Y): number_of_clicks, number_of_bbp, total_whistles_duration. 
# Explanatory variables (X): acoustic, fishing_net, behavior, beacon, net, number.

# What are the H0/ H1 hypotheses ?
# H0 : No influence of any of the explanatory variables on a dependant one.
# H1 : Influence of an explanatory variable on a dependent one.

##################### DATA EXPLORATION ################################
# Y Outlier detection
par(mfrow=c(2,3))
boxplot(acoustic.dta$total_whistles_duration, col='red', 
        ylab='total_whistles_duration')
boxplot(acoustic.dta$number_of_bbp, col='red', 
        ylab='number_of_bbp')
boxplot(acoustic.dta$number_of_clicks, col='red', 
        ylab='number_of_clicks')

dotchart(acoustic.dta$total_whistles_duration, pch=16, 
         xlab='total_whistles_duration', col='red')
dotchart(acoustic.dta$number_of_bbp, pch=16, 
         xlab='number_of_bbp', col='red')
dotchart(acoustic.dta$number_of_clicks, pch=16, 
         xlab='number_of_clicks', col='red')

# Y distribution
par(mfrow=c(2,3))
hist(acoustic.dta$total_whistles_duration, col='red', breaks=8,
        xlab='total_whistles_duration', ylab='number')
hist(acoustic.dta$number_of_bbp, col='red', breaks=8,
        xlab='number_of_bbp', ylab='number')
hist(acoustic.dta$number_of_clicks, col='red', breaks=8,
        xlab='number_of_clicks', ylab='number')

qqnorm(acoustic.dta$total_whistles_duration, col='red', pch=16)
qqline(acoustic.dta$total_whistles_duration)
qqnorm(acoustic.dta$number_of_bbp, col='red', pch=16)
qqline(acoustic.dta$number_of_bbp)
qqnorm(acoustic.dta$number_of_clicks, col='red', pch=16)
qqline(acoustic.dta$number_of_clicks)

shapiro.test(acoustic.dta$total_whistles_duration)
shapiro.test(acoustic.dta$number_of_bbp)
shapiro.test(acoustic.dta$number_of_clicks)
# p-values are significant => they do not follow normal distributions
# will need a transformation or the use of a glm model

# X Number of individuals per level
summary(factor(acoustic.dta$acoustic))
summary(factor(acoustic.dta$fishing_net))
summary(factor(acoustic.dta$behavior))
summary(factor(acoustic.dta$beacon))
summary(factor(acoustic.dta$net))
table(factor(acoustic.dta$acoustic),factor(acoustic.dta$fishing_net))
table(factor(acoustic.dta$acoustic),factor(acoustic.dta$behavior))
table(factor(acoustic.dta$behavior),factor(acoustic.dta$acoustic))
ftable(factor(acoustic.dta$fishing_net), factor(acoustic.dta$behavior), factor(acoustic.dta$acoustic))
# => unbalanced, no big deal but will need more work (no orthogonality):
# Effects can depend on the order of the variables 

# => Beacon and net have modalities with <10 individuals => analysis impossible
# => They will be treated apart from the rest as they are likely to be biased


##################### STATISTICAL MODELLING ###########################
### Model tested
# LM: Linear model (residual hypothesis: normality, homoscedasticity, independant)
# GLM: Generalized linear model (residual hypothesis: homoscedasticity, independant)
# NB : Negative Binomial model (usually, when overdispersion with GLM)
# ZINB: Zero inflated negative binomial model (residual hypothesis: homoscedasticity, independant
# using number as an offset (more dolphins => more signals)

# beacon and net explanatory variables could not be tested in models 
# as they contain information already present in "fishing_net" which is more 
# interesting to keep for our study. They will be treated after 
# (using kruskall-Wallis non-parametric test)
# fishing_net, behavior and acoustic where tested with their interactions.
# If a variable is it in a model, it is because it had no significant effect.

par(mfrow=c(1,1))

### Model for whistles
# Residual hypotheses not verified for LM
# Overdipsersion when using GLM (negative binomial)
# Using ZINB:
zero.whi <- zeroinfl(total_whistles_duration ~ 
                      acoustic + fishing_net + behavior + offset(log(number)), 
                    data=acoustic.dta, dist='negbin')
nb.whi <- glm.nb(total_whistles_duration ~ 
                     acoustic + fishing_net + behavior + offset(log(number)), 
                   data=acoustic.dta)
# comparison ZINB VS NB model
vuong(zero.whi, nb.whi)  #(if p-value<0.05 then first model in comparison is better)
mod.whi <- zero.whi # => zeroinflated model is indeed better suited
car::Anova(mod.whi, type=3)
shapiro.test(residuals(mod.whi)) # H0 : normality -> not rejected if p>0.05
dwtest(mod.whi) # H0 -> independent if p>0.05 (autocorrelation if p<0.05)
bptest(mod.whi) # H0 -> homoscedasticity if p<0.05
# No normality but we do not need it


### Model for BBP
# No normality of residuals for LM
# overdispersion with GLM quasipoisson
#try with glm NB:
mod.bbp <- glm.nb(number_of_bbp ~ acoustic + fishing_net + behavior 
                  + offset(log(number)),
                  data=acoustic.dta)
car::Anova(mod.bbp, type=3)
dwtest(mod.bbp) # H0 -> independent if p>0.05 (autocorrelation if p<0.05)
bptest(mod.bbp) # H0 -> homoscedasticity if p<0.05
# Normality not needed in GLM, hypotheses verified !
mod.bbp$deviance/mod.bbp$df.residual 
# slight underdispersion, not improved with ZINB so we keep this


### Model for clicks
# Using NB model:
mod.cli <- glm.nb(number_of_clicks ~ acoustic + fishing_net + acoustic:fishing_net + offset(log(number)), 
               data=acoustic.dta)
car::Anova(mod.cli, type=3)
shapiro.test(residuals(mod.cli)) # H0 : normality -> cannot be rejected if p > 0.05
dwtest(mod.cli) # H0 -> independent if p>0.05 (autocorrelation if p<0.05)
bptest(mod.cli) # H0 -> homoscedasticity if p<0.05
# Normality not needed in GLM, hypotheses verified !
mod.cli$deviance/mod.cli$df.residual
# slight overdispersion. (ZINB does not clearly improve results so we keep this)

# FYI1: Comparison of combination of explanatory variables between models
# were compared based on BIC criterion. 
# The model with the lowest BIC was kept (and is the one shown)

# FYI2: log(number of dolphin per group) does have an effect on data but we have
# no interest in investigating it, that is why we use it as an offset.


##################### Boxplots and comparisons ##################### 
### Functions to compute stats
computeLetters <- function(temp, category) {
  test <- multcomp::cld(object = temp$emmeans,
                        Letters = letters)
  myletters_df <- data.frame(category=test[,category],
                             letter=trimws(test$.group))
  colnames(myletters_df)[1] <- category
  return(myletters_df)
}

computeStats <- function(data, category, values, two=NULL, three=NULL) {
  my_sum <- data %>%
    group_by({{category}}, {{two}}, {{three}}) %>% 
    summarise( 
      n=n(),
      mean=mean({{values}}),
      sd=sd({{values}})
    ) %>%
    mutate( se=sd/sqrt(n))  %>%
    mutate( ic=se * qt((1-0.05)/2 + .5, n-1))
  return(my_sum)
}

barPlot <- function(dta, signif, category, old_names, new_names, fill=NULL, size=5,
                    height, xname="", colours="black", legend_title="", legend_labs="",ytitle=""){
  if (!is.null(signif)){colnames(signif)[1] <- "use"}
  
  dta %>%
    mutate(use=fct_relevel({{category}}, old_names)) %>%
    ggplot(aes(x=use, y=mean, group={{fill}}, fill={{fill}},color={{fill}})) +
    {if(length(colours)==1)geom_point(color=colours, position=position_dodge(.5))}+
    {if(length(colours)==2)geom_point(position=position_dodge(.5), show.legend = FALSE)}+
    {if(length(colours)==2)scale_color_manual(values=colours, name=legend_title, labels=legend_labs)}+
    scale_x_discrete(breaks=old_names,
                     labels=new_names)+
    ylab(ytitle)+
    xlab(xname)+
    theme_classic()+ theme(text=element_text(size=12))+
    {if(!is.null(signif))geom_text(data=signif, aes(label=letter, y=height), size=size,
                                   colour="black", position=position_dodge(.5))}+
    geom_errorbar(aes(x=use, ymin=mean-ic, ymax=mean+ic), position=position_dodge(.5), width=.1, show.legend = FALSE)
  
}

####Introducing variables averaged per dolphins ####
# since we introduced an offset, variables can be divided by the number of dolphins
acoustic.dta$whistling_time_per_dolphin <- acoustic.dta$total_whistles_duration/acoustic.dta$number
acoustic.dta$BBPs_per_dolphin <- acoustic.dta$number_of_bbp/acoustic.dta$number
acoustic.dta$clicks_per_dolphin <- acoustic.dta$number_of_clicks/acoustic.dta$number

#### Fishing net  ####
# whistles
table <- cld(emmeans(mod.whi, pairwise~fishing_net, adjust="tukey"), Letters = letters)
myletters_df <- data.frame(fishing_net=table$fishing_net,
                           letter = trimws(table$.group))
barPlot(computeStats(acoustic.dta, fishing_net, whistling_time_per_dolphin/375), # 375 bins = 1 sec
        myletters_df, fishing_net, 
        old_names = c("SSF","F"), new_names = c("Absent", "Present"),
        xname="Presence/Asence of fishing net", height=.5, 
        ytitle="Mean whistling time per dolphin per min")

# BBP
table <- cld(emmeans(mod.bbp, pairwise~fishing_net, adjust="tukey"), Letters = letters)
myletters_df <- data.frame(fishing_net=table$fishing_net,
                           letter = trimws(table$.group))
barPlot(computeStats(acoustic.dta, fishing_net, BBPs_per_dolphin),
        myletters_df, fishing_net, 
        old_names = c("SSF","F"), new_names = c("Absent", "Present"),
        xname="Presence/Asence of fishing net", height=.6, 
        ytitle="Mean number of BBPs per dolphin per min")

# Clicks
table <- cld(emmeans(mod.cli, pairwise~fishing_net, adjust="tukey"), Letters = letters)
myletters_df <- data.frame(fishing_net=table$fishing_net,
                           letter = trimws(table$.group))
barPlot(computeStats(acoustic.dta, fishing_net, clicks_per_dolphin),
        myletters_df, fishing_net, 
        old_names = c("SSF","F"), new_names = c("Absent", "Present"),
        xname="Presence/Asence of fishing net", height=100, 
        ytitle="Mean number of clicks per dolphin per min")


#### Acoustic plots  ####
# Whistles
table <- cld(emmeans(mod.whi, pairwise~acoustic, adjust="tukey"), Letters = letters)
myletters_df <- data.frame(acoustic=table$acoustic,letter = trimws(table$.group))
barPlot(computeStats(acoustic.dta, acoustic, whistling_time_per_dolphin/375),
        myletters_df, acoustic, height=0.65, ytitle="Mean whistling time per dolphin per min",
        old_names = c("AV","AV+D","D","D+AP","AP"),
        new_names = c("BEF","BEF+DUR","DUR", "DUR+AFT", "AFT"),
        xname="Activation sequence")

# BBPs
table <- cld(emmeans(mod.bbp, pairwise~acoustic, adjust="tukey"), Letters = letters)
myletters_df <- data.frame(acoustic=table$acoustic,letter = trimws(table$.group))
barPlot(computeStats(acoustic.dta, acoustic, BBPs_per_dolphin),
        myletters_df, acoustic, height=1.2, ytitle="Mean number of BBPs per dolphin per min",
        old_names = c("AV","AV+D","D","D+AP","AP"),
        new_names = c("BEF","BEF+DUR","DUR", "DUR+AFT", "AFT"),
        xname="Activation sequence")

# Clicks
table <- cld(emmeans(mod.cli, pairwise~acoustic, adjust="tukey"), Letters = letters)
myletters_df <- data.frame(acoustic=table$acoustic,letter = trimws(table$.group))
barPlot(computeStats(acoustic.dta, acoustic, clicks_per_dolphin),
        myletters_df, acoustic, height=155, ytitle="Mean number of clicks per dolphin per min",
        old_names = c("AV","AV+D","D","D+AP","AP"),
        new_names = c("BEF","BEF+DUR","DUR", "DUR+AFT", "AFT"),
        xname="Activation sequence")


#### Interaction fishing_net:acoustic plots  ####
# Whistles
letters_df <- computeLetters(emmeans(mod.whi, pairwise~fishing_net:acoustic, adjust="tukey"), 
                             "fishing_net")
letters_df$acoustic <- computeLetters(emmeans(mod.whi, pairwise~fishing_net:acoustic, adjust="tukey"), 
                                      "acoustic")$acoustic
letters_df <- letters_df[, c("acoustic","fishing_net","letter")]
letters_df$letter <- gsub(" ", "", letters_df$letter)
barPlot(computeStats(acoustic.dta, fishing_net, whistling_time_per_dolphin/375, two=acoustic),
        NULL, acoustic, fill=fishing_net,
        old_names = c("AV","AV+D","D","D+AP","AP"), ytitle="Mean whistling time per dolphin per min",
        new_names = c("BEF","BEF+DUR","DUR", "DUR+AFT", "AFT"),
        xname="Activation sequence", height=c(.95,.95,.95,1,.95,1,.95,1,1,1), 
        colours=c("#E69F00","#999999"), size=5,
        legend_title="Fishing net", legend_labs=c("Present", "Absent"))

# BBPs
letters_df <- computeLetters(emmeans(mod.bbp, pairwise~fishing_net:acoustic, adjust="tukey"), 
                             "fishing_net")
letters_df$acoustic <- computeLetters(emmeans(mod.bbp, pairwise~fishing_net:acoustic, adjust="tukey"), 
                                      "acoustic")$acoustic
letters_df <- letters_df[, c("acoustic","fishing_net","letter")]
letters_df$letter <- gsub(" ", "", letters_df$letter)
barPlot(computeStats(acoustic.dta, fishing_net, BBPs_per_dolphin, two=acoustic),
        NULL, acoustic, fill=fishing_net,
        old_names = c("AV","AV+D","D","D+AP","AP"), ytitle="Mean number of BBPs per dolphin per min",
        new_names = c("BEF","BEF+DUR","DUR", "DUR+AFT", "AFT"),
        xname="Activation sequence", height=c(1.65,1.65,1.72,1.65,1.72,1.65,1.65,1.72,1.72,1.72), 
        colours=c("#E69F00","#999999"), size=5, 
        legend_title="Fishing net", legend_labs=c("Present", "Absent"))

# Clicks
letters_df <- computeLetters(emmeans(mod.cli, pairwise~fishing_net:acoustic, adjust="tukey"), 
                             "fishing_net")
letters_df$acoustic <- computeLetters(emmeans(mod.cli, pairwise~fishing_net:acoustic, adjust="tukey"), 
                                      "acoustic")$acoustic
letters_df <- letters_df[, c("acoustic","fishing_net","letter")]
letters_df$letter <- gsub(" ", "", letters_df$letter)
barPlot(computeStats(acoustic.dta, fishing_net, clicks_per_dolphin, two=acoustic),
        NULL, acoustic, fill=fishing_net,
        old_names = c("AV","AV+D","D","D+AP","AP"), ytitle="Mean number of clicks per dolphin per min",
        new_names = c("BEF","BEF+DUR","DUR", "DUR+AFT", "AFT"),
        xname="Activation sequence", height=c(180,180,187,187,180,187,180,180,187,187), 
        colours=c("#E69F00","#999999"), size=5, 
        legend_title="Fishing net", legend_labs=c("Present", "Absent"))

#### Behaviour plots ####
# Whistles
table <- cld(emmeans(mod.whi, pairwise~behavior, adjust="tukey"), Letters = letters)
myletters_df <- data.frame(behavior=table$behavior,letter = trimws(table$.group))
barPlot(computeStats(acoustic.dta, behavior, whistling_time_per_dolphin/375),
        myletters_df, behavior, height=0.75, ytitle="Mean whistling time per dolphin per min",
        old_names = c("CHAS", "DEPL", "SOCI"),
        new_names = c("Foraging", "Travelling", "Socialising"),
        xname="Behaviours of dolphins")

# BBPs
# real effect measured in model
table <- cld(emmeans(mod.bbp, pairwise~behavior, adjust="tukey"), Letters = letters)
myletters_df <- data.frame(acoustic=table$behavior,letter = trimws(table$.group))
barPlot(computeStats(acoustic.dta, behavior, BBPs_per_dolphin),
        myletters_df, behavior, height=1.2, ytitle="Mean number of BBPs per dolphin per min",
        old_names = c("CHAS", "DEPL", "SOCI"),
        new_names = c("Foraging", "Travelling", "Socialising"),
        xname="Behaviours of dolphins")

# Clicks
# no significant effect in click statistical model so all the same letters
myletters_df <- data.frame(behavior=unique(acoustic.dta$behavior),
                           letter = rep("a",length(unique(acoustic.dta$behavior))))
barPlot(computeStats(acoustic.dta, behavior, clicks_per_dolphin),
        myletters_df,
        behavior, old_names = c("CHAS", "DEPL", "SOCI"),
        new_names = c("Foraging", "Travelling", "Socialising"),
        xname="Behaviours of dolphins", height=150,
        ytitle="Mean number of clicks per dolphin per min")

#### Nets plots + KW analysis #### 
# Whistles
#KW test
kruskal.test(acoustic.dta$whistling_time_per_dolphin ~ acoustic.dta$net)
# p<0.05 so post-hoc
kruskalmc(acoustic.dta$whistling_time_per_dolphin, acoustic.dta$net)
# DIY : letters
myletters_df <- data.frame(net=c("SSF", "chalut_blanc", "chalut_vert", "tremail", "grand_filet"),
                           letter = c("a","ad","bd","cd","a"))
barPlot(computeStats(acoustic.dta, net, whistling_time_per_dolphin/375),
        NULL,
        net, old_names = c("SSF", "chalut_blanc", "chalut_vert", "tremail", "grand_filet"),
        new_names = c("Absent", "Nylon trawl net", "PE trawl net", "Nylon gill net", "Long nylon gill net"),
        xname="Fishing nets", height=.6,
        ytitle="Mean whistling time per dolphin per min")+
    theme(axis.text.x=element_text(size=8.5))

# BBPs
#KW test
kruskal.test(acoustic.dta$BBPs_per_dolphin ~ acoustic.dta$net)
# p<0.05 so post-hoc
kruskalmc(acoustic.dta$BBPs_per_dolphin, acoustic.dta$net)
# DIY : letters
myletters_df <- data.frame(net=c("SSF", "chalut_blanc", "chalut_vert", "tremail", "grand_filet"),
                           letter = c("a","a","a","a","a"))
barPlot(computeStats(acoustic.dta, net, BBPs_per_dolphin),
        NULL,
        net, old_names = c("SSF", "chalut_blanc", "chalut_vert", "tremail", "grand_filet"),
        new_names = c("Absent", "Nylon trawl net", "PE trawl net", "Nylon gill net", "Long nylon gill net"),
        xname="Fishing nets", height=.8,
        ytitle="Mean number of BBPs per dolphin per min")+
  theme(axis.text.x=element_text(size=8.5))

# Clicks
#KW test
kruskal.test(acoustic.dta$clicks_per_dolphin ~ acoustic.dta$net)
# p<0.05 so post-hoc
kruskalmc(acoustic.dta$clicks_per_dolphin, acoustic.dta$net)
# DIY : letters
myletters_df <- data.frame(net=c("SSF", "chalut_blanc", "chalut_vert", "tremail", "grand_filet"),
                           letter = c("ae","ad","bd","cd","e"))
barPlot(computeStats(acoustic.dta, net, clicks_per_dolphin),
        NULL,
        net, old_names = c("SSF", "chalut_blanc", "chalut_vert", "tremail", "grand_filet"),
        new_names = c("Absent", "Nylon trawl net", "PE trawl net", "Nylon gill net", "Long nylon gill net"),
        xname="Fishing nets", height=120,
        ytitle="Mean number of clicks per dolphin per min")+
  theme(axis.text.x=element_text(size=8.5))


#### Beacon plots + KW analysis (letters not shown for readability) ####
# Whistles
#KW test
kruskal.test(acoustic.dta$whistling_time_per_dolphin ~ acoustic.dta$beacon)
names = computeStats(acoustic.dta, beacon, whistling_time_per_dolphin/375)["beacon"]
barPlot(computeStats(acoustic.dta, beacon, whistling_time_per_dolphin/375),
        NULL,
        beacon, old_names = unlist(names), new_names = unlist(names),
        xname="Signals from bio-inspired beacon", height=0.9, size=3,
        ytitle="Mean whistling time per dolphin per min")+
  theme(axis.text.x=element_text(size=8))+
  scale_x_discrete(guide=guide_axis(n.dodge = 2))
# NC stands for "Unknown". Corresponding to categories where the beacon was not turned on yet ('BEF')

# BBPs
#KW test
kruskal.test(acoustic.dta$BBPs_per_dolphin ~ acoustic.dta$beacon)
names = computeStats(acoustic.dta, beacon, whistling_time_per_dolphin/375)["beacon"]
barPlot(computeStats(acoustic.dta, beacon, BBPs_per_dolphin),
        NULL,
        beacon, old_names = unlist(names), new_names = unlist(names),
        xname="Signals from bio-inspired beacon", height=0.5, size=3,
        ytitle="Mean number of BBPs per dolphin per min")+
      theme(axis.text.x=element_text(size=8))+
      scale_x_discrete(guide=guide_axis(n.dodge = 2))
# NC stands for "Unknown". Corresponding to categories where the beacon was not turned on yet ('BEF')

# Clicks
#KW test
kruskal.test(acoustic.dta$clicks_per_dolphin ~ acoustic.dta$beacon)
names = computeStats(acoustic.dta, beacon, whistling_time_per_dolphin/375)["beacon"]
barPlot(computeStats(acoustic.dta, beacon, clicks_per_dolphin),
        NULL,
        beacon, old_names = unlist(names), unlist(names),
        xname="Signals from bio-inspired beacon", height=150, size=3,
        ytitle="Mean number of clicks per dolphin per min")+
  theme(axis.text.x=element_text(size=8))+
  scale_x_discrete(guide=guide_axis(n.dodge = 2))
# NC stands for "Unknown". Corresponding to categories where the beacon was not turned on yet ('BEF')

#### WHY NOT: Number plots ####
# Whistles
numb_stats_w <- computeStats(acoustic.dta, number, total_whistles_duration/375)
numb_stats_w[is.na(numb_stats_w)] <- 0
numb_stats_w$number <- as.factor(numb_stats_w$number)

numb_stats_w %>%
  ggplot(aes(x=number, y=mean, group=1)) +
  geom_errorbar(aes(x=number, ymin=mean-ic, ymax=mean+ic), 
                color="red", width=.1, show.legend = FALSE)+
  geom_point() + geom_line() +
  theme_classic() + theme(text=element_text(size=12)) +
  ylab("Mean whistling time per min")+
  xlab("Number of dolphins in group")

# BBPs
numb_stats_b <- computeStats(acoustic.dta, number, number_of_bbp)
numb_stats_b[is.na(numb_stats_b)] <- 0
numb_stats_b$number <- as.factor(numb_stats_b$number)

numb_stats_b %>%
  ggplot(aes(x=number, y=mean, group=1)) +
  geom_errorbar(aes(x=number, ymin=mean-ic, ymax=mean+ic), 
                color="red", width=.1, show.legend = FALSE)+
  geom_point() + geom_line() +
  theme_classic() + theme(text=element_text(size=12)) +
  ylab("Number of BBPs per min")+
  xlab("Number of dolphins in group")

# Clicks
numb_stats_c <- computeStats(acoustic.dta, number, number_of_clicks)
numb_stats_c[is.na(numb_stats_c)] <- 0
numb_stats_c$number <- as.factor(numb_stats_c$number)

numb_stats_c %>%
  ggplot(aes(x=number, y=mean, group=1)) +
  geom_errorbar(aes(x=number, ymin=mean-ic, ymax=mean+ic), 
                color="red", width=.1)+
  geom_point() + geom_line() +
  theme_classic() + theme(text=element_text(size=12)) +
  ylab("Mean number of clicks per min")+
  xlab("Number of echolocation clicks in group")

