export AUC, gini

using DataFrames: by, DataFrame, sort!
using CategoricalArrays: CategoricalVector
using SortingLab: fsortperm

function AUC_plot_data(score, target::CategoricalVector;  kwargs...)
    @assert length(levels(target)) == 2
    _AUC_plot_data(score, 2 .- target.refs; kwargs...)
end

AUC_plot_data(score, target;  kwargs...) = _AUC_plot_data(score, target;  kwargs...)

function _AUC_plot_data(pred, target;  plotauc = false)
    @assert length(pred) == length(target)
    s = fsortperm(pred)
    ts = @view target[s]
    FN = cumsum(ts)
    TP = sum(ts) .- FN
    TPR = TP ./ (TP .+ FN)

    FP = (1:length(ts)) .- FN
    TN = (length(target) - sum(target)) .- FP
    FPR = FP ./(FP .+ TN)

    sum((TPR[1:end-1] .+ TPR[2:end]) .* diff(FPR) ./ 2)

    FPR, TPR
end

"""
    AUC(score, target; plotauc = false)

Return the AUC. To generate a plot set `plotauc=true`.
"""
function AUC(score, target; kwargs...)
    cu, cutarget = AUC_plot_data(score, target; kwargs...)
    sum((cutarget[2:end] .+ cutarget[1:end-1]) .* diff(cu) ./ 2)
end

"""
    gini(score, target; plotauc = false)

Return the `gini = (AUC - 0.5)/0.05 = 2AUC - 1`. AUC is the area under the curve while gini
is the ratio of (AUC minus the area of the bottom triangle) vs (Area of upper triangle).

For AUC a random model has AUC = 0.5 but for gini a random model has gini = 0.0

To generate a plot set `plotauc=true`.
"""
function gini(score, target; plotauc = false)
    auc = AUC(score,target; plotauc = plotauc)
    2*auc-1
end
