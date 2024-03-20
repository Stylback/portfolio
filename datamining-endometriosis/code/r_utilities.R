## Libraries
require(plyr)
require(dplyr)
require(tidyr)
require(FactoMineR)
require(factoextra)
require(reshape2)
require(ggplot2)
setwd("../data/")

## Describes relative prevalence of endometriosis for select categories
## Based on this article: https://datascienceplus.com/using-mca-and-variable-clustering-in-r-for-insights-in-customer-attrition/
describe_categories <- function(df_subset, rep_times, endo_prop) {
    options(repr.plot.width = 14, repr.plot.height = 14)

    counts <- c()
    for (col in colnames(df_subset)[2:ncol(df_subset)]){
        count <- ddply(df_subset, .(df_subset[[col]], df_subset$Diagnosis), "nrow")
        names(count) <- c("class", "Diagnosis", "count")
        counts <- as.data.frame(rbind(counts, count))
    }

    variables <- rep(names(df_subset[2:ncol(df_subset)]), times = c(rep_times))

    df_details <- as.data.frame(cbind(variables, counts))
    endo_prevalence <- dcast(df_details, variables + class ~ Diagnosis, value.var = "count")
    endo_prevalence <- subset(endo_prevalence, endometriosis != "NA") # nolint: object_usage_linter.
    endo_prevalence <- subset(endo_prevalence, no_diagnosis != "NA") # nolint: object_usage_linter.
    endo_prevalence <- subset(endo_prevalence, class != "NA")
    endo_prevalence <- subset(endo_prevalence, class != "")
    endo_prevalence <- mutate(endo_prevalence, relative_prevalence = round((endometriosis / (no_diagnosis + endometriosis) * 100) - endo_prop, 1)) # nolint: object_usage_linter.

    categories <- as.data.frame(paste(endo_prevalence$variable, endo_prevalence$class))
    names(categories) <- c("Category")
    prevalence_table <- as.data.frame(cbind(categories, endo_prevalence))

    ## Plot options
    fig <- ggplot(prevalence_table, aes(Category, relative_prevalence, color = relative_prevalence < 0)) + # nolint: object_usage_linter.
                    geom_segment(aes(x = reorder(Category, relative_prevalence), xend = Category, y = 0, yend = relative_prevalence), linewidth = 1.1, alpha = 0.7) +
                    theme(legend.position = "none") +
                    xlab("Category") +
                    ylab("Relative prevalence [%]") +
                    ggtitle("Relative prevalence of endometriosis, \ncompared to overall average") +
                    geom_label(aes(label = relative_prevalence, size = NULL)) +
                    coord_flip()

    fig <- fig + theme(text = element_text(size = 20))

    return(fig)
}

## Multiple Correspondence Analysis (MCA)
## Based on this article: https://www.sthda.com/english/articles/31-principal-component-methods-in-r-practical-guide/114-mca-multiple-correspondence-analysis-in-r-essentials/
visualize_mca <- function(res_mca) {

    # To get a straight line from no_diagnosis to endometriosis in the first orthogonal plane
    endo_x <- res_mca$quali.sup$coord[1]
    endo_y <- res_mca$quali.sup$coord[3]
    no_diag_x <- res_mca$quali.sup$coord[2]
    no_diag_y <- res_mca$quali.sup$coord[4]
    xx <- c(endo_x, no_diag_x)
    yy <- c(endo_y, no_diag_y)
    slope <- diff(yy) / diff(xx)
    intercept <- yy[1] - slope * xx[1]
    perpendicular_slope <- (-1 / slope)

    options(repr.plot.width = 14, repr.plot.height = 14)
    fig <- fviz_mca_var(res_mca, col.var = "cos2",
                axes = c(1, 2),
                map = "symmetrical",
                select.var = list(name = NULL, cos2 = 15, contrib = NULL),
                gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
                col.quali.sup = "#5720f0",
                shape.var = 19,
                label = "all",
                labelsize = 5,
                repel = TRUE,
                ggtheme = theme_minimal()
                ) + theme(
                    text = element_text(size = 20),
                    axis.title = element_text(size = 25),
                    axis.text = element_text(size = 15)) + geom_abline(
                        intercept = intercept, slope = slope,
                        linetype = "dotted", color = "#5720f0") + geom_abline(
                        intercept = intercept, slope = perpendicular_slope,
                        linetype = "dotted", color = "#5720f0") + annotate(
                            "text", x = endo_x * 10, y = endo_y * 10, label = "associated") + annotate(
                                "text", x = -endo_y * 10, y = endo_x * 10, label = "disassociated")
    return(fig)
}