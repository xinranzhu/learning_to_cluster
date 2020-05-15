using TensorOperations
α=randn()
A=randn(5,5,5,5,5,5)
B=randn(5,5,5)
C=randn(5,5,5)
D=zeros(5,5,5)
@tensor begin
    D[a,b,c] = A[a,e,f,c,f,g]*B[g,b,e] + α*C[c,a,b]
    E[a,b,c] := A[a,e,f,c,f,g]*B[g,b,e] + α*C[c,a,b]
end