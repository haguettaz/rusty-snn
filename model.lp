\ Model model1
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  8 x1 + x2
Subject To
 c0: x1 + 2 x2 >= -14
 c1: - 4 x1 - x2 <= -33
 c2: 2 x1 + x2 <= 20
Bounds
 x1 free
 x2 free
Generals
 x2
End
