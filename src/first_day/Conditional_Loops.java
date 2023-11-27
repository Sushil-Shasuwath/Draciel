package first_day;

import java.util.Scanner;

public class Conditional_Loops {
    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        int n = in.nextInt();
        int digit = in.nextInt();
//        int num1 = in.nextInt();
//        int num2 = in.nextInt();
//
//        int largest = Math.max(num1, num2);
//        System.out.println(largest);

//        char ch = in.next().trim().charAt(0);

//        System.out.println(in.next().trim()+in.next()+in.next());

//        int a = 0;
//        int b = 1;
//        int count = 2;
//        while(count<=n){
//            int temp = b;
//            b+=a;
//            a = temp;
//            count++;
//        }
//        System.out.println(b);

//        int res = fib(n);
//        System.out.println(res);

//        //palindrome
//        int temp = n;
//        int sum=0,rem,count=0;
//        while(n>0){
//            rem = n%10;
//            if(rem==digit){
//                count++;
//            }
//            sum=(sum*10)+rem;
//            n/=10;
//        }
//        System.out.println("Reversed number"+sum);
//        System.out.println(temp==sum?"It's a palindrome":"It's not a palindrome");
//        System.out.println("Count of "+digit+" : "+count);

        int div = n/digit;
        System.out.println(div);

    }
//    static int fib(int n)
//    {
//        if (n <= 1)
//            return n;
//        return fib(n - 1) + fib(n - 2);
//    }
}
