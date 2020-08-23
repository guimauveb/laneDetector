#include <iostream>

/* Construct the polynom :
    * Iterates through the array starting from the end since polyfit_boost returns 
    * the polynomial coefficients in decremental powers. 
*/
template <typename T>
void poly1d(const std::vector<T> polyfit)
{
    int degree = polyfit.size() - 1;
    for (int i = degree; i >= 0; i--) {

        if (polyfit[i-1] < 0) {
            if (i == degree)
                std::cout << polyfit[i] << "x - ";
            else if (i == 0) {
                std::cout << abs(polyfit[i]);
                break;
            }
            else if (i == 1)
                std::cout << abs(polyfit[i]) << " x - ";
            else
                std::cout << abs(polyfit[i]) << " x" << i << " - ";
        }

        else if (polyfit[i-1] >= 0) {
            if (i == degree)
                std::cout << polyfit[i] << "x + ";
            else if (i == 0) {
                std::cout << abs(polyfit[i]);
                break;
            }
            else if (i == 1)
                std::cout << abs(polyfit[i]) << " x + ";
            else
                std::cout << abs(polyfit[i]) << " x" << i << " + ";
            }

    }
    std::cout << std::endl;
}

/* Evaluate the polynomial at x */
template <typename T>
int poly1d_eval(const std::vector<T> polyfit, const int x)
{
    int degree = polyfit.size() - 1;
    int eval;
    for (int i = degree; i > 0; i--){
        eval = polyfit[i]*pow(x, degree);
    }
    /* Add constant */
    eval += polyfit[0];
    // std::cout << "test at x = " << x << " | eval = " << eval << std::endl;
    
    return eval;
}
