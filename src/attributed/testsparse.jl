I = [1;2; 2; 3; 1]
J = [2;3; 5; 4; 5]
V = [100;200; 250; 300; 500]

A = sparse(I,J,V)
rows = rowvals(A)
vals = nonzeros(A)
m, n = size(A)
for j = 1:n
    println("nzrange: ", nzrange(A, j))
   for i in nzrange(A, j)
      row = rows[i]
      val = vals[i]
      println("row: ", row)
      println("val: ", val)
      println("ACTUAL VAL:", A[row, j])
      # perform sparse wizardry...
   end
end



