#include <iostream>
#include <vector>
#include <cmath>
#include <string>

struct Point {
    double x, y;
    
    Point(double x = 0, double y = 0) : x(x), y(y) {}
    
    double get() const {
        return(x, y);
    }
    std::string getString() const {
        return "(" + std::to_string(x) + ", " + std::to_string(y) + ")";
    }
};


std::pair<Point, Point> calculateOrthogonalPoints(const Point& A, const Point& B, double distance) {
    double dx = B.x - A.x;
    double dy = B.y - A.y;
    double magnitude = std::sqrt(dx*dx + dy*dy);
    double dx_normalized = dx / magnitude;
    double dy_normalized = dy / magnitude;
    Point orthogonalVectorAbove(dy_normalized, -dx_normalized);
    Point orthogonalVectorBelow(-dy_normalized, dx_normalized);
    Point pointAbove(B.x + distance * orthogonalVectorAbove.x, B.y + distance * orthogonalVectorAbove.y);
    Point pointBelow(B.x + distance * orthogonalVectorBelow.x, B.y + distance * orthogonalVectorBelow.y);
    
    return {pointAbove, pointBelow};
};

Point calculateMidwayPoint(const Point& A, const Point& B, const Point& Anchor, double distance) {
    double xh = (A.x + B.x) / 2;
    double yh = (A.y + B.y) / 2;
    double xd = (xh - Anchor.x);
    double yd = (yh - Anchor.y);
    double magnitude = std::sqrt(xd*xd + yd*yd);
    double xd_normalized = xd / magnitude;
    double yd_normalized = yd / magnitude;
    return Point(xh + distance * xd_normalized, yh + distance * yd_normalized);
};


extern "C" {

    void calculatePolygons(const double* X, const double* Y, int N, double distance, double** polygonsX, double** polygonsY) {
        std::vector<std::pair<Point, Point>> midwayPointsList(N-2);

        for(int i = 1; i < N - 1; ++i) {
            // Calculate orthogonal points for the line segment between (X[i-1], Y[i-1]) and (X[i], Y[i])
            Point A(X[i-1], Y[i-1]), B(X[i], Y[i]), C(X[i+1], Y[i+1]);
            std::pair<Point, Point> orthogonalPointsAB = calculateOrthogonalPoints(A, B, distance);
            std::pair<Point, Point> orthogonalPointsCB = calculateOrthogonalPoints(C, B, distance);

            // Calculate the midway point between the orthogonal points
            Point midwayPointAB = calculateMidwayPoint(orthogonalPointsAB.first, orthogonalPointsCB.second, B, distance);
            Point midwayPointCB = calculateMidwayPoint(orthogonalPointsCB.first, orthogonalPointsAB.second, B, distance);

            // Store the points
            midwayPointsList[i-1] = std::make_pair(midwayPointAB, midwayPointCB);
        }

        // Allocate memory for the output arrays
        *polygonsX = (double*)malloc((N - 3) * sizeof(double) * 6);
        *polygonsY = (double*)malloc((N - 3) * sizeof(double) * 6);

        for(int i = 0; i < N - 3; ++i) {
            (*polygonsX)[i * 6 + 0] = midwayPointsList[i].first.x;
            (*polygonsY)[i * 6 + 0] = midwayPointsList[i].first.y;
            (*polygonsX)[i * 6 + 1] = X[i+1];
            (*polygonsY)[i * 6 + 1] = Y[i+1];
            (*polygonsX)[i * 6 + 2] = midwayPointsList[i].second.x;
            (*polygonsY)[i * 6 + 2] = midwayPointsList[i].second.y;
            (*polygonsX)[i * 6 + 3] = midwayPointsList[i+1].second.x;
            (*polygonsY)[i * 6 + 3] = midwayPointsList[i+1].second.y;
            (*polygonsX)[i * 6 + 4] = X[i+2];
            (*polygonsY)[i * 6 + 4] = Y[i+2];
            (*polygonsX)[i * 6 + 5] = midwayPointsList[i+1].first.x;
            (*polygonsY)[i * 6 + 5] = midwayPointsList[i+1].first.y;
        }
    }
}
