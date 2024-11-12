---
layout: page
title: "Social Disparity in Impacts of Climate Disasters (2023)"
description: Using a data-driven approach and leveraging machine learning models to identify trends in how climate disasters impact different demographic groups in the U.S. 
img: assets/img/1_cover.jpg
importance: 1
category: data
related_publications: false
---

<div class="row">
    <div class="col-sm-8 mt-3 mt-md-0 mx-auto d-block">
        {% include figure.liquid loading="eager" path="assets/img/project_1/1.png" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    Aftermath of Hurricane Harvey in Texas in 2017 (Image Credit: <a href="https://www.vox.com/science-and-health/2017/8/29/16219788/hurricane-harvey-recovery-joplin">Vox</a>)
</div>

### Introduction

With the planet's changing climate comes an ever more extreme, frequent and unpredictable landscape of climate disasters. The ways that climate change will affect natural disasters are complex and manifold, including the increased frequency of droughts and heat waves, and increased intensity of storms with increasing global surface temperatures [1]. As of December 8, there have been 25 confirmed climate disaster events in 2023 each with over $1 billion loss in the United States, compared to the annual average of 18 events over the past 5 years (2018-2022) [2]. While these climate disasters impact every population group in the U.S., not everyone will be affected equally. A plethora of studies show that in the U.S., marginalized populations, including people of color, immigrants, low-income households, and people for whom English is not their native language, are more vulnerable to the impacts of climate change [3]. This increased vulnerability can be attributed to a multitude of factors, including reduced access to resources such as transportation and healthcare. People’s geographic location can also shape their experience of climiate impacts. For example, a 2017 study found that Black people were 52% more likely than white people to live in high heat-risk areas, while Hispanics were 21% more likely, making these racial minority groups more susceptible heat-related health impacts [8].

​

The aftermath of climate disaster events such as hurricanes, flooding, wildfires, and heat waves are particularly revealing of how climate change impacts and social injustice are tightly intertwined. Disadvantaged communities are particularly at risk because they may live in subpar housing more prone to power outages and damage, and lack access to resources to relocate or recover after a disaster. Communities of color are also less likely to receive adequate protection against disasters or a prompt response in case of emergencies. These inequalities translate to unevenly distributed climate disaster impacts, that include lost homes, livelihoods, and lives, as well as a range of short- and long-term health and economic outcomes on both a personal and community level. One devastating example of social disparity in climate disaster impacts was with Hurricane Katrina in 2005. More than half of the 1,200 people who died were Black and 80 percent of the homes that were destroyed belonged to Black residents, partially because these poorer neighborhoods had received less government funding for flood protection. After the hurricane, white neighborhoods were prioritized in initial plans for rebuilding, even if they experienced less significant flooding. A study examining long-term racial risparities in health and economic conditions among Hurricane Katrina survivors found that Black survivors were more likely to report hurricane-related problems with personal health, emotional well-being, and finances [5].

 <div class="row">
    <div class="col-sm-10 mt-3 mt-md-0 mx-auto d-block">
        {% include figure.liquid loading="eager" path="assets/img/project_1/2.png" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    (Image Credit: <a href="https://www.ncei.noaa.gov/access/billions/">NOAA</a>)
</div>
​
As we face mounting climate challenges, we must couple our understanding of climate justice with social justice in order to make sure resources are distributed to people who need it most, especially in the wake of climate disasters. Furthermore, the people who experience the most vulnerability to climate change are often those with not only one, but multiple marginalized identities [4]. This is described by the framework of intersectionality, which describes how different dimensions of an individual's identity intersect to determine one’s experience in the world and how intersecting forces of privilege and oppression are at work in a given society. In the status quo, climate change is yet anohter contributor of social injustice. However, because climate justice and social justice are inextricably linked, there is also opportunity to work towards solutions for climate change that can also promote social equality. According to Naomi Klein, "climate change [...] could become a galvanizing force for humanity, leaving us all not just safer from extreme weather, but with societies that are safer and fairer in all kinds of other ways as well" [7]. The use of data to understand and predict the impact of climate change is critical to supporting these aims.

There is a large amount of work being done in predicting climate impacts, but only a small subset of this research explicitly address how these impacts will be distributed across different social groups. One such related work is a recent 2021 United States Environmental Protection Agency (EPA) study predicts on the degree to which four socially vulnerable populations may be more exposed to 6 of the highest impacts of climate change [6]. In this report, they projected that low income and African-American individuals will face higher impacts of climate change for all 6 impacts, including health impacts of extreme temperatures and property loss due to coastal and inland flooding. Another finding is that with 2°C (3.6°F) of global warming, Hispanic and Latino individuals are 43% more likely to live in areas with the highest projected reductions in labor hours (i.e. in weather-exposed industries, such as construction and agriculture) due to extreme temperatures. This report is an key milestone in understanding the impacts of climate change on different American populations, but there are still many facets yet to be explored.

This project applies a data-driven approach to investigate disparities in the impacts of climate disasters across across demographics such as race and income level in the United States. In particular, this work aims to draw insights about who has been most impacted by extreme weather events and to quantify these impacts across a wide range of data sources and measures, from the Federal Emergency Management Agency (FEMA’s) projected national climate risk index by county to survey data on hurricane survivors’ reported impacts, with the goal of informing more effective decision making and distribution of resources in the face of this growing crisis. Whenever possible, multiple demographics will be included in a single analysis to reveal insights about the impacts of climate change on the most vulnerable populations. 7 machine learning models with a range of have different will be applied to reveal insights from these datasets. While there have been studies that apply data science and machine learning on social vulnerability and climate change, such as the aforementioned EPA study [6], this project uniquely applies a range of machine learning models on a data sources, in order to build rich and multifaceted insights on inequality in impact of climate disasters.



### Quick summary of research questions and findings:
1. How has the frequency and total cost of major climate disasters changed over the past 20 years?

    In the last 3 years (2020-2021), there was an annual average of 20 billion-dollar natural disasters in the U.S., costing a total of $152 billion per year. This is compared to the 2010s, which had an annual average of 13 billion-dollar climate events at an average cost of $97 billion per year, and the 2000s, averaging 6.7 billion-dollar climate events with an average cost of $60 billion per year (CPI-adjusted). As of December 8, there have been 25 confimed billion-dollar weather/climate disaster events in 2023.
​
2. Which socioeconomic populations are most affected by climate change impacts based on where they are geographically located?

    Using FEMA's National Risk Index and the CDC's Social Vulnerability Index by county, k-means clustering showed that geographical locations with large low-income, black, and Hispanic populations also tend have highest climate impacts. Regression analysis revealed the largest increases in climate risk for percent population increase in the following relationships: the total climate risk vs. percentage Asian population, heat wave risk vs. percentage Asian population (these may be attributed to the smaller overall Asian-American population), heat wave risk vs. percentage African-American population, total climate risk vs. percentage African-American population, and total climate risk vs. percentage Hispanic population.

3. How is an individual's inability to meet essential expenses following climate disasters related to or predicted by socioeconomic factors? 

    Association rule mining was used to find associations between demographic groups and reported impacts of hurricanes Harvey, Irma, and Nate in 2017 with data from the Survey of Trauma, Resilience, and Opportunity Among Neighborhoods in the Gulf (STRONG). This model found that the strongest associations in the data were between an individual being aged 51-75, female, and/or experiencing high adversity due to the hurricane impacts—and being black. In this analysis, high adversity corresponds to experiencing at least 3 hurricane impacts, including not meeting all essential expenses, not paying the full rent or mortgage, being evicted, and not having adequate food.

4.  How is an individual's experience of job loss following a climate disaster related to or predicted by socioeconomic factors?

    Association rule mining was used to find associations between demographic groups and reported impacts of hurricanes Harvey, Irma, and Nate in 2017 with data from the Survey of Trauma, Resilience, and Opportunity Among Neighborhoods in the Gulf (STRONG). The strongest associations related to job loss revealed that job loss in the aftermath of these hurricanes is highly correlated with individuals that are 51-75, black, and male.

5. How is an individual's ability to fully recover following a climate disaster related to or predicted by socioeconomic factors?

    From running the naive Bayes algorithm on data from the Harvey Anniversary survey revealed that the most important predictor of recovery from Hurricane Harvey was home damage due to the hurricane, but income and race predictors were the next most important predictors, even more so than experiencing reduced work hours due to the storm. Furthermore, the conditional probability of recovery is 1.5 times as high given someone is white compared to if they are African American, and the conditional probability of not recovering is 1.5 times as high given someone is living in poverty compared to not living in poverty.

6. How is an individual's mental well-being following a climate disaster related to or predicted by socioeconomic factors?

    Mental well-being variables of depression score and PTSD score were used in the association rule mining analysis, but they were not included in the top associations computer by the model. In future work, the results of the model could be filtered on those that include those mental health measures or other models could be used to analyze these impacts.

7. How are different socially vulnerable groups impacted by climate disasters?

    The k-means clustering and regression analysis on social vulnerability and climate risk by geographical location indicated that black populations may suffer more from heat wave related risks, compared to Hispanic populations.

8. How are people with multiple marginalized identities are impacted by climate disasters (more heavily than those with just one marginalized identity)?

    Several results support the understanding that people with more than one marginalized identity often face the greatest risks of climate change, due to intersecting forces of privilege and oppression. For example, the association rule mining model reveals how multiple socioeconomic vulnerabilities such as being both female and black, and experiencing high adversity from Hurricanes Harvey, Nate, and Imra, are strongly connected.

<!-- 6. How is an individual's experience of displacement following a climate disaster related to or predicted by socioeconomic factors?

    This question was not answered during the course of this project, but the same models used in this project applied to data from the STRONG or Harvey Anniversary surveys, could be used to answer this question in future work -->


<!-- 10. Using existing predicted climate disaster risk data, what are the future impacts of climate disasters across different demographics?

    These were not predicted due to an inability to find the necessary data to perform this analysis during the course of this project, although this would be important future work. -->

##### References:

1. USGS U.S. Geological Survey. "How can climate change affect natural disasters?" <https://www.usgs.gov/faqs/how-can-climate-change-affect-natural-disasters>
2. NOAA National Centers for Environmental Information (NCEI) (2023). "U.S. Billion-Dollar Weather and Climate Disasters (2023)." <https://www.ncei.noaa.gov/access/billions/>
3. Cho, R. (2020). "Why Climate Change is an Environmental Justice Issue." Columbia Cliate School. <https://news.climate.columbia.edu/2020/09/22/climate-change-environmental-justice/>
4. EPA (2023). "Climate Change and Children’s Health and Well-Being in the United States." U.S. Environmental Protection Agency. <https://www.epa.gov/cira/climate-change-and-childrens-health-report>
5. Toldson, I. A., Ray, K., Hatcher, S. S., & Straughn Louis, L. (2011). Examining the Long-Term Racial Disparities in Health and Economic Conditions Among Hurricane Katrina Survivors: Policy Implications for Gulf Coast Recovery. Journal of Black Studies, 42(3), 360-378. <https://doi.org/10.1177/0021934710372893>
6. EPA (2021). "Climate Change and Social Vulnerability in the United States: A Focus on Six Impacts." U.S. Environmental Protection Agency. <https://www.epa.gov/cira/social-vulnerability-report>
7. Klein, M. (2014). "This Changes Everything: Capitalism Vs. The Climate." Simon and Schuster.