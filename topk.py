def quicksort(v, left, right):
     if left < right:
         p = parttion(v, left, right)
         if p=k:
             return v[k]
         elif p<k:
            quicksort(v, left, p-1)
         else:
            quicksort(v, p+1, right)
