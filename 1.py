alist=[2,1,3,5,1,3]
# for i in range(len(nums) - 1):  # 遍历 len(nums)-1 次
#         minIndex = i
#         for j in range(i + 1, len(nums)):
#             if nums[j] < nums[minIndex]:  # 更新最小值索引
#                 minIndex = j
#                 nums[i], nums[minIndex] = nums[minIndex], nums[i] # 把最小数交换到前面
#
# print(nums)


length = len(alist)
# for i in range(length - 1):
#         # i表示比较多少轮
#     for j in range(1, length-i):
#             # j表示每轮比较的元素的范围，因为每比较一轮就会排序好一个元素的位置，
#             # 所以在下一轮比较的时候就少比较了一个元素，所以要减去i
#         if alist[j-1] > alist[j ]:
#             alist[j-1], alist[j ] = alist[j ], alist[j-1]
# print(alist)
# for i in range(1,9):
#    for j in range(i, 9):
#       print("*",end='')
#    print('\r')


# class people:
#     # 定义基本属性
#     name = ''
#     age = 0
#     # 定义私有属性,私有属性在类外部无法直接进行访问
#     __weight = 0
#
#     # 定义构造方法
#     def __init__(self, n, a, w):
#         self.name = n
#         self.age = a
#         self.__weight = w
#
#     def speak(self, a1):
#         print("%s 说: 我 %d 岁。" % (a1, self.age))
#
#
# # 实例化类
# p = people('runoob', 10, 999)
# print(p)
# class FooParent(object):
#     def __init__(self):
#         self.parent = 'I\'m the parent.'
#         print('Parent')
#
#     def bar(self, message):
#         print("%s from Parent" % message)
#
#
# class FooChild(FooParent):
#     def __init__(self):
#         # super(FooChild,self) 首先找到 FooChild 的父类（就是类 FooParent），然后把类 FooChild 的对象转换为类 FooParent 的对象
#         super(FooChild, self).__init__()
#         print('Child')
#
#     def bar(self, message):
#         super(FooChild, self).bar(message)
#         print('Child bar fuction')
#         print(self.parent)
#
#
# if __name__ == '__main__':
#     fooChild = FooChild()
#     fooChild.bar('HelloWorld')


# def bubbleSort(arr):
#     n = len(arr)
#
#     # 遍历所有数组元素
#     for i in range(1, n):
#
#         # Last i elements are already in place
#         for j in range(i,0,-1):
#
#             if arr[j] < arr[j - 1]:
#                 arr[j], arr[j - 1] = arr[j - 1], arr[j]
#

list1 = [64, 34, 25, 12, 22, 11, 90]
# if 1:
#     n = len(list)
#     for i in range(1, n):
#         # 后一个元素和前一个元素比较
#         # 如果比前一个小
#         if list[i] < list[i - 1]:
#             # 将这个数取出
#             temp = list[i]
#             # 保存下标
#             index = i
#             # 从后往前依次比较每个元素
#             for j in range(i - 1, -1, -1):
#                 # 和比取出元素大的元素交换
#                 if list[j] > temp:
#                     list[j + 1] = list[j]
#                     index = j
#                 else:
#                     break
#             # 插入元素
#             list[index] = temp
#
# def quick_sort(alist, start, end):
# #     if start >= end:
# #         return
# #     low = start
# #     high = end
# #     mid = alist[low]
# #
# #     while low < high:
# #         while low < high and mid < alist[high]:
# #             # 从右边开始找，如果元素小于基准，则把这个元素放到左边
# #             high -= 1
# #         alist[low] = alist[high]
# #
# #         while low < high and mid > alist[low]:
# #             # 从左边开始找，如果元素大于基准，则把元素放到右边
# #             low += 1
# #         alist[high] = alist[low]
# #
# #     # 循环退出，low==high，把基准元素放到这个位置
# #     alist[low] = mid
# #
# #     # 递归调用，重新排列左边的和右边的序列
# #     quick_sort(alist, start, low - 1)
# #     quick_sort(alist, low + 1, end)
# #
# #
# #
# def f(n):
#     if n ==1:
#         return 1
#     elif n == 2:
#         return 2
#     else:
#         s1=f(n-1)+f(n-2)
#         return s1
#
# def d(n):
#     if n == 1:
#         return 1
#     elif n == 2:
#         return 2
#     else:
#         a = 1
#         b = 2
#         # 这个数列从第3项开始，每一项都等于前两项之和。
#         for i in range(2, n):
#
#             c = a+b
#             a = b  #把后面一项赋值给前面的变量
#             b = c  #把得出的结果赋值给下一个需要加的变量
#         return c
#
# print(d(5))


# n = 999
# b = []  # 存储余数
# while True:  # 一直循环，商为0时利用break退出循环
#         s = n // 2  # 商
#         y = n % 2  # 余数
#         b = b + [y]  # 每一个余数存储到b中
#         if s == 0:
#             break  # 余数为0时结束循环
#         n = s

# def power(base,exponent):
#     res = 1
#     while exponent:
#         if exponent & 1:  # 判断当前的最后一位是否为1，如果为1的话，就需要把之前的幂乘到结果中。
#             res = res*base
#         base = base*base  # 一直累乘，如果最后一位不是1的话，就不用了把这个值乘到结果中，但是还是要乘。
#         exponent = exponent >> 1
#     return res
#
#
# a = power(3,999)
# str1 = '0123456789'
#
# def main():
#     test_str = 'hello world'
#     test_list = list(test_str)
#     print(test_list)
#
#
# if __name__ == '__main__':
#     main()
#

# for x in range(5):
#     #print(1)
#     if x == 6:
#         print(x)
#         break
# else:
#     print(x)
#     print("执行else....")
#
#
########################################################################
# n = [[1,1,1,1,1,1,1,1,1,1],
#      [1,0,0,1,0,0,0,1,0,1],
#      [1,0,0,1,0,0,0,1,0,1],
#      [1,0,0,0,0,1,1,0,0,1],
#      [1,0,1,1,1,0,0,0,0,1],
#      [1,0,0,0,1,0,0,0,0,1],
#      [1,0,1,0,0,0,1,0,0,1],
#      [1,0,1,1,1,0,1,1,0,1],
#      [1,1,0,0,0,0,0,0,0,1],
#      [1,1,1,1,1,1,1,1,1,1]]
#
# dirs = [
#     lambda x,y:(x+1,y),
#     lambda x,y:(x-1,y),
#     lambda x,y:(x,y-1),
#     lambda x,y:(x,y+1)]
#
#
# def maze_path(x1,y1,x2,y2):
#     stack = []
#     stack.append((x1,y1))
#     while(len(stack)>0):
#         curNode = stack[-1]
#         if curNode[0]==x2 and curNode[1]==y2:
#             for p in stack:
#                 print(p)
#             return True
#         for dir in dirs:
#             nextNode =dir(curNode[0], curNode[1])
#             if n[nextNode[0]][nextNode[1]] == 0:
#                 stack.append(nextNode)
#                 n[nextNode[0]][nextNode[1]] = 2
#                 break
#         else:
#             #n[nextNode[0]][nextNode[1]] = 2
#             stack.pop()
#             #print(nextNode[0],nextNode[1])
#     else:
#         print("没有路！")
#         return False
#
#
# maze_path(1,1,8,9)



########################################################################
class Node():
    def __init__(self,item):
        self.item = item
        self.next = None

def qiancha(n):
    A=Node(n[0])  # 因为是头插，所以头结点每次都在往前移动
    for i in n[1:]:
        node = Node(i)
        node.next = A  # 前插：新的结点在原来的结点前面
        A = node  # 循环，每次替换掉头部的结点
    return A

def weicha(n):
    A = Node(n[0])
    B = A  # 因为是尾插，所以先把A链表复制一份，不然头结点找不到
    for i in n[1:]:
        node = Node(i)
        B.next = node  # 尾插：新的结点在原来的结点后面
        B = node  # 循环，每次替换掉尾部的结点
    return A


ls = weicha([1,3,5])
























