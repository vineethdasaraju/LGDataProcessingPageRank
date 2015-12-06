pagerankVD <- function(data, inputs, outputs) {
    inputs <- convert.exprs(substitute(inputs))
    outputs <- convert.atts(substitute(outputs))
    gla <- GLA(pagerankVD::page_rank_vd)
    Aggregate(data, gla, inputs, outputs)
}
