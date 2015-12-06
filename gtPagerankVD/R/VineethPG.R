VineethPG <- function(data, inputs, outputs) {
    inputs <- convert.exprs(substitute(inputs))
    outputs <- convert.atts(substitute(outputs))
    gla <- GLA(vineeth-pg::ConnectedComponents)
    Aggregate(data, gla, inputs, outputs)
}
