
using DataFrames
using LinearAlgebra
using FreqTables
using StatsBase


Base.@kwdef struct kNN
    k::Int64
    p::Float64
    data::DataFrame
    labels::DataFrame
    classify::Function
    kNN(k, p, data, labels) = new(k, p, data, labels, 
        (x::DataFrameRow) -> begin
            ftd = Dict(
                freqtable(labels[map(
                    s::DataFrameRow -> norm(collect(s) - collect(x), p), eachrow(data)
                ) .<= kDistance(data, x, k, p), 
                :][!, 1]
            ))
            argmax(ftd)
        end
    )
end

function kDistance(data::DataFrame, x::DataFrameRow, k::Int64, p::Float64)
    sort(map(s::DataFrameRow -> norm(collect(s) - collect(x), p), eachrow(data)))[k]
end

function dataframe_classify(f::T, x::DataFrame) where T<:Function
    DataFrame(class=map(x_ -> f(x_), eachrow(x)))
end

"""
    acc(f, x, y)

Get accuracy of the kNN classifier.

f<:Function - kNN classifier.classify function.
x::DataFrame - input data.
y::DataFrame - labels.

Example of usage:
```
> dataset = DataFrame(shuffle(eachrow(Iris().dataframe)))

> trn_data = dataset[1:120, :]
> test_data = dataset[121:end, :]

> classifier = kNN(3, 2.0, trn_data[:,1:4], DataFrame(class=trn_data[:,5]));
> acc(classifier.classify, test_data[:,1:4], DataFrame(class=test_data[:,5]));
```
"""
function acc(f::T, x::DataFrame, y::DataFrame) where T<:Function 
    mean(Int.(dataframe_classify(f, x) .== y).class)
end

function leave_one_out(dataset::DataFrame, k::Int64, p::Float64)
    results::Vector{Bool} = []
    for i::Int64=1:size(dataset)[1]
        tmp_d::DataFrame = deepcopy(dataset)
        leaved::DataFrame = DataFrame(tmp_d[i, :])
        deleteat!(tmp_d, i)
        classifier = kNN(k, p, tmp_d[:,1:4], DataFrame(class=tmp_d[:,5]))
        append!(results, [classifier.classify(leaved[1, 1:4]) == leaved[1,5]])
    end
    mean(results)
end

leave_one_out(dataset::DataFrame, k::Int64, p::Int64) = leave_one_out(dataset,k, Float64(p))