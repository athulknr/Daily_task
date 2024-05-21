def main():
   
    list1 = set(map(int, input("Enter numbers for the first list (separated by spaces): ").split()))

   
    list2 = set(map(int, input("Enter numbers for the second list (separated by spaces): ").split()))

    
    intersection = list1 & list2
    union = list1 | list2
    difference1 = list1 - list2
    difference2 = list2 - list1
    
   
    print("Intersection:", intersection)
    print("Union:", union)
    print("Difference (list1 - list2):", difference1)
    print("Difference (list2 - list1):", difference2)

if __name__ == "__main__":
    main()