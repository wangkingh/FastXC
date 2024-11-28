/******************************************************************************
 * This code is my collection and minor revision of several subroutines to
 * calculate the azimuth back-azimuth great circle and distance in km for two
 * points located on the surface of earth.
 *
 * Definition:
 * Since it is code for seismology, we try to follow the traditional definition.
 * Suppose we have one EVENT point located on (evlo,evla) in degree.
 * and another       STATION point located on (stlo,stla) in degree.
 * then we can define the az baz and gcarc and distkm as follows:
 * az     ===== Azimuth: angle counted clockwise from North for vector
 *              from event to station. in degree.
 * baz    ====  Back-Azimuth: angle counted clockwise from North for vector from
 *              station to event. in degree
 * gcarc  ====  great circle linking the two points, in degree
 * distkm ====  length of the line linking these two points,in KM
 *
 * There are many method to calculate these parameters, the most simple and so that
 * less accurate method is using the spherecial triangle formula to get these parameter,
 * with or without the lon/lat correction of earth flat factor.
 *
 * We here collect two subroutine to calculate the az/baz/gcarc/distkm from two
 * points, and this is aways called Forward Method in geodesic.
 *
 * 1. The method used in SAC
 *    1.the az/baz/azimuth is calculated using Bullen's equation in
 *      A introduction to seismogical theory 4th section b 10.2b
 *    2.the distkm is calculated using Rudoe's method.
 *    this method is used in sac and gsac as well as gmt4.xx in gmt_map.c
 *
 * 2. The method of Vincenty
 *    It is said by author of GMT the Vincenty method is more accurate than Rudoe's method.
 *    And this method is used in GMT5.0 as well as the GeographicLib by Charles Karney
 *    I Unwrap this from GMT5.01
 *
 * There do exist many other method, but maybe this two is enough for routine calculation
 * of az/baz/gcarc/distkm. Other options such as the back-projection and midpoint problem
 * should be incoprated when I have time.
 *
 * Reference:
 * 1.A introduction to seismogical theory 4th section b 10.2b
 * 2.Santos and de Souza,
 *   Applications of Rudoe's Formula in Long Seismic Surface Wave Path Determination,RBC,2005
 * 3. Karney 2013 paper
 *
 * Note:
 * 1.Sac(Version 101.5c) use earth
 *   rad = 6378.160;    Earth Radius
 *   fl  = 0.00335293;  Earth Flattening
 *   But actually in WGS 84 the rad is 6378.137km and fl=1.0/298.257223563
 *   We use WGS84 as the earthmodel so the result may be a little bit different from sac
 * 2.sac use float to store the evla/evla/stlo/stla, we want to use double to minimus the
 *   computering error within the subroutine.So I write a wrapper function which convert
 *   float to double first then convert double to float back,hoho.
 *   This is also good for quick switch between different method.
 * 3.Since the Vincenty method use a different formula very special for Geodesic, the
 *   great circle may be not right. but the distance is saiy to be very accurate.
 * 4.also I add some hack code to avoid bad results caused by precision lost in Rudoe
 *   Method. very simple just to make sure the ouput az/baz is within [0,360)
 * 5.May be the Rudoe method is well enough. I compare the results for a huge loop, the
 *   difference of distance in Km is not very large compared to Vincenty method.
 *
 *
 * Reminder
 * the  = theta_eq = evla
 * phe  = phi_eq   = evlo
 * ths  = theta_st = stla
 * phs  = phi_st   = stlo
 * dist = distkm
 * xdeg = gcarc
 * az   = azimuth  , source->station vector wrt N clockwise
 * baz  = backazimuth  station->source vector wrt N clockwise
 *
 *
 * History:
 * 1. unwraped by wangwt from sac-101.3b src  @2010??
 * 2. documented by wangwt@20130821
 * 3. add some hack codes and a wrapper function by wangwt@20130906
 * Todo:
 * 1. the az/baz/gcarc should be split out to be single subroutine so that we can have
 *    more combination of the az/baz/gcarc with distkm
 * 2. since most of the time we do not need that high precision result, maybe we should
 *    make out one code which using just the GeoGraphic2GeoCentric convention and basic
 *    speretriangle formula to calculate the az/baz/gcarc/midpoint/backproj/bearingangle etc
 *    that may be beautiful. and I believe there should someone who have finished already.
 *
 *
 * last update by wangwt@20130906
 ***************************************************************************************/
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "hddistance.h"

void distkm_az_baz_Vincenty(double evlo, double evla, double stlo, double stla, double *badgcarc, double *outaz, double *outbaz, double *distkm);
void distkm_az_baz_Rudoe(double evlo, double evla, double stlo, double stla, double *gcarc, double *az, double *baz, double *distkm);
void hd_distaz(float evlo, float evla, float stlo, float stla, float *gcarc, float *az, float *baz, float *distkm)
{
    /* this is a wrapper for sac's float format */
    double tmpevlo, tmpevla, tmpstlo, tmpstla, tmpgcarc, tmpaz, tmpbaz, tmpdistkm;
    tmpevlo = (double)evlo;
    tmpevla = (double)evla;
    tmpstlo = (double)stlo;
    tmpstla = (double)stla;
    /* call routine */
    distkm_az_baz_Rudoe(tmpevlo, tmpevla, tmpstlo, tmpstla, &tmpgcarc, &tmpaz, &tmpbaz, &tmpdistkm);
    // printf("DOUBLE gcarc %f az %f baz %f distkm %f\n",tmpgcarc,tmpaz,tmpbaz,tmpdistkm);
    /* now convert back to float */
    *gcarc = (float)tmpgcarc;
    *az = (float)tmpaz;
    *baz = (float)tmpbaz;
    *distkm = (float)tmpdistkm;
}

/* Latitudes (which are input as geographic latitudes) vary from 0 to 90 in degree,
 * + for North and - for south.
 * Longitudes vary from -180 to +180 in degree,+ for East and - for West.
 * OR
 * Longitudes vary from 0 to 360(to the East, with 0=Greenwich Meridian);
 * See Wiki for the explanation for geographic and geocentric latitude as well as co-latitude.
 *
 * azimuth     is the vector event->station with respect to N clockwise.
 * backazimuth is the vector station->event with respect to N clockwise.
 wangwt@20110624 */

void distkm_az_baz_Rudoe(double evlo, double evla, double stlo, double stla, double *gcarc, double *az, double *baz, double *distkm)
{

    /* constants for radian-degree conveting. M_PI is defined in math.h */
    double DEG2RAD = M_PI / 180.0;
    double RAD2DEG = 180.0 / M_PI;

    /* earth constant of WGS84. note it is a little different from sac  */
    double EARTHR = 6378.137;             /* Earth Radius,major axis,eg radius at the equator */
    double EARTHFL = 1.0 / 298.257223563; /* Earth Flattening factor,fl=(a-b)/a */

    /* ellipsoid constant for earth. When convert geographical latitude into geocentrical latitude,
     *  use the formula tan(geoCentLat)=ONEMEC2*tan(geoGraphLat) */
    double EC2, ONEMEC2;
    EC2 = 2.0 * EARTHFL - EARTHFL * EARTHFL; /* EC2=e^2=(a^2-b^2)/a^2 */
    ONEMEC2 = 1.0 - EC2;                     /* ONEMEC2=one minus EC2=b^2/a^2 */
#ifdef nouse
    double EPS;
    EPS = 1.0 + EC2 / ONEMEC2; /* EPS=a^2/b^2 >1 */
#endif

    /* Latitude/Longitude of event and station in radian */
    double evlaRad, evloRad, stlaRad, stloRad;
    /* for geocentric latitude for event and station in radian */
    double temp;
    double evlaGeoCent, stlaGeoCent;
    /* spherecial coordinates for event and station,see Bullen's book */
    double a, b, c, d, e, f, g, h;         /* event   spherical coordinates */
    double a1, b1, c1, d1, e1, f1, g1, h1; /* station spherical coordinates */
    /* ss=sine(s),sc=cos(s),sd=deg(s)=gcarc */
    double ss, sc, sd;

    /* a hack if the two points are exactly same points, set all to zero */
    if (evlo == stlo && evla == stla)
    {
        *gcarc = 0.0;
        *az = 0.0;
        *baz = 0.0;
        *distkm = 0.0;
        /* how to exist this void  */
        return;
    }
    /************* az/baz/gcarc calculation using Bullen's formula  *******/
    /* Bullen's formula is not stable when latitude of exactly zero 0.0 */
    temp = evla;
    if (temp == 0.0)
        temp = 1.0e-10;
    /* convert degree to rad for event location */
    evlaRad = DEG2RAD * temp; // evla
    evloRad = DEG2RAD * evlo; // evlo
    /* Must convert from geographic to geocentric coordinates in order
     * to use the spherical triangle equations.  This requires a latitude
     * correction given by: 1-EC2=1-2*FL+FL*FL */
    if (evla == 90.0 || evla == -90.0)
    {
        evlaGeoCent = evla * DEG2RAD;
    }
    else
    {
        evlaGeoCent = atan(ONEMEC2 * tan(evlaRad));
    }
    /* now calculate the A to H for bullen formula,from sac */
    d = sin(evloRad);
    e = -cos(evloRad);
    f = -cos(evlaGeoCent);
    c = sin(evlaGeoCent);
    a = f * e;
    b = -f * d;
    g = -c * e;
    h = c * d;
    /* now same correction for station */
    temp = stla;
    if (temp == 0.0)
        temp = 1e-10;
    stlaRad = temp * DEG2RAD;
    stloRad = stlo * DEG2RAD;
    /* convert geographical latitude into geocentric latitude */
    if (stla == 90 || stla == -90)
    {
        stlaGeoCent = stla * DEG2RAD;
    }
    else
    {
        stlaGeoCent = atan(ONEMEC2 * tan(stlaRad));
    }
    /* for station a - h */
    d1 = sin(stloRad);
    e1 = -cos(stloRad);
    f1 = -cos(stlaGeoCent);
    c1 = sin(stlaGeoCent);
    a1 = f1 * e1;
    b1 = -f1 * d1;
    g1 = -c1 * e1;
    h1 = c1 * d1;
    sc = a * a1 + b * b1 + c * c1; /* vector dot product return cos(gcarc) */
    /* for gcarc */
    sd = 0.5 * sqrt((pow(a - a1, 2.0) + pow(b - b1, 2.0) + pow(c - c1, 2.0)) * (pow(a + a1, 2.0) + pow(b + b1, 2.0) + pow(c + c1, 2.0)));
    /* now get the gcarc and convert to degree */
    temp = atan2(sd, sc) * RAD2DEG;
    /* make sure gcarc > 0. We do not check <0 and eps as az/baz
     * because gcarc will not go to 360.0 that frequently  */
    if (temp < 0.0)
        temp += 360.0;
    *gcarc = temp;
    /* for azimuth */
    ss = pow(a1 - d, 2.0) + pow(b1 - e, 2.0) + pow(c1, 2.0) - 2.0;
    sc = pow(a1 - g, 2.0) + pow(b1 - h, 2.0) + pow(c1 - f, 2.0) - 2.0;
    /* for azimuth radian to degree */
    temp = atan2(ss, sc) * RAD2DEG;
    /* hack code to aviod precision lost. if az = -1e-14, it should
     * be zero. we set eps to 0.0000001 see enough  wangwt@20130906 */
    if (temp < 0 && fabs(temp) < 1e-8)
    { // hack to avoid az=360
        temp = 0.0;
    }
    if (temp < 0.0)
        temp += 360.0;
    *az = temp;
    /* swap event and station for back azimuth calculation */
    ss = pow(a - d1, 2.0) + pow(b - e1, 2.0) + pow(c, 2.0) - 2.0;
    sc = pow(a - g1, 2.0) + pow(b - h1, 2.0) + pow(c - f1, 2.0) - 2.0;
    temp = atan2(ss, sc) * RAD2DEG;
    if (temp < 0 && fabs(temp) < 1e-8)
    { // hack to avoid az=360
        temp = 0.0;
    }
    if (temp < 0.0)
        temp += 360.0;
    *baz = temp;

    /***************** End of az/baz/gcarc ************************************/

    /********* Rudoe's Alogirithm for Distance Calculation  *******************/
    /* Constants for Rudoe's formula  */
    /******************************************************************
     Below are parameters for Rudoe's algorithm for calculating Distance
     in KM.or in meter if you set earth radius in meter.
    *******************************************************************/
    /* t=theta,p=phi for point 1 and 2 */
    double t1, p1, t2, p2;
    /* for Rudoe Algorithm */
    double sinthk, costhk, tanthk, sinthi, costhi, tanthi;
    /*  Rudoe Algorithm continue  */
    double a12, cosa12, sina12, a12top, a12bot;
    /*  Rudoe Algorithm continue  */
    double el, e2, e3;
    double al, dl, du;
    double c0, c2, c4;
    double v1, v2;
    double z1, z2;
    double x2, y2;
    double e1p1, sqrte1p1;
    double u1, u2, u1bot, u2bot, u2top;
    double b0;
    double pdist;
    /***************************************************
     constant for Rudoe's formula, see Santos's paper.
     cxx is the polynomial coefficient
     c00=1,      c01=1/4,   c02=-3/64,   c03=5/256
     c21=-1/8,   c22=13/32, c23=-15/1024
     c42=-1/256, c43=3/1024
    ***************************************************/
    double c00 = 1.;
    double c01 = 1.0 / 4.00;
    double c02 = -3.0 / 64.0;
    double c03 = 5.0 / 256.0;
    double c21 = -1.0 / 8.0;
    double c22 = 13.0 / 32.0;
    double c23 = -15.0 / 1024.0;
    double c42 = -1.0 / 256.0;
    double c43 = 3.0 / 1024.0;
    /* - Now compute the distance between the two points using Rudoe's
     *   formula given in GEODESY, section 2.15(b).
     *   (There is some numerical problem with the following formulae.
     *   If the station is in the southern hemisphere and the event in
     *   in the northern, these equations give the longer, not the
     *   shorter distance between the two locations.  Since the
     *   equations are fairly messy, the simplist solution is to reverse
     *   the meanings of the two locations for this case.)
     *   This means we should assume the event at southern and station at northern part */

    if (stlaRad > 0.0)
    {                 /* station located in the northern hemisphere */
        t1 = stlaRad; /* Point1 is station, Point2 is event */
        p1 = stloRad;
        t2 = evlaRad;
        p2 = evloRad;
        /* special attention at the poles to avoid atan2 troubles and division by zero. */
        // thk stand for the event ????
        if (evla == 90.0)
        {
            costhk = 0.0;
            sinthk = 1.0;
            tanthk = FLT_MAX; // FLT_MAX is defined in float.h
        }
        else if (evla == -90.0)
        {
            costhk = 0.0;
            sinthk = -1.0;
            tanthk = -FLT_MAX;
        }
        else
        {
            costhk = cos(t2);
            sinthk = sin(t2);
            tanthk = sinthk / costhk;
        }
        /* special attention at the poles to avoid atan2 troubles and division by zero. */
        if (stla == 90.0)
        {
            costhi = 0.0;
            sinthi = 1.0;
            tanthi = FLT_MAX;
        }
        else if (stla == -90.0)
        {
            costhi = 0.0;
            sinthi = -1.0;
            tanthi = -FLT_MAX;
        }
        else
        {
            costhi = cos(t1);
            sinthi = sin(t1);
            tanthi = sinthi / costhi;
        }
    }
    else
    {                 /* station is in the southern hemisphere, swap event and station */
        t1 = evlaRad; /* Now point1 is event point2 is station */
        p1 = evloRad;
        t2 = stlaRad;
        p2 = stloRad;
        /* special attention at the poles to avoid atan2 troubles and division by zero. */
        if (stla == 90.0)
        {
            costhk = 0.0;
            sinthk = 1.0;
            tanthk = FLT_MAX;
        }
        else if (stla == -90.0)
        {
            costhk = 0.0;
            sinthk = -1.0;
            tanthk = -FLT_MAX;
        }
        else
        {
            costhk = cos(t2);
            sinthk = sin(t2);
            tanthk = sinthk / costhk;
        }
        /* special attention at the poles to avoid atan2 troubles and division by zero. */
        if (evla == 90.0)
        {
            costhi = 0.0;
            sinthi = 1.0;
            tanthi = FLT_MAX;
        }
        else if (evla == -90.0)
        {
            costhi = 0.0;
            sinthi = -1.0;
            tanthi = -FLT_MAX;
        }
        else
        {
            costhi = cos(t1);
            sinthi = sin(t1);
            tanthi = sinthi / costhi;
        }
    } // end of swap station and event

    /* now begin to use the Rudoe's method to calculate the distance */
    el = EC2 / ONEMEC2;
    e1 = 1.0 + el;
    al = tanthi / (e1 * tanthk) + EC2 * sqrt((e1 + pow(tanthi, 2.0)) / (e1 + pow(tanthk, 2.0)));
    dl = p1 - p2;
    a12top = sin(dl);
    a12bot = (al - cos(dl)) * sinthk;
    /* Rewrote these three lines with help from trig identities.  maf 990415 */
    a12 = atan2(a12top, a12bot);
    cosa12 = cos(a12);
    sina12 = sin(a12);
    e1 = el * (pow(costhk * cosa12, 2.0) + pow(sinthk, 2.0));
    e2 = e1 * e1;
    e3 = e1 * e2;
    c0 = c00 + c01 * e1 + c02 * e2 + c03 * e3;
    c2 = c21 * e1 + c22 * e2 + c23 * e3;
    c4 = c42 * e2 + c43 * e3;
    v1 = EARTHR / sqrt(1. - EC2 * pow(sinthk, 2.0));
    v2 = EARTHR / sqrt(1. - EC2 * pow(sinthi, 2.0));
    z1 = v1 * (1. - EC2) * sinthk;
    z2 = v2 * (1. - EC2) * sinthi;
    x2 = v2 * costhi * cos(dl);
    y2 = v2 * costhi * sin(dl);
    e1p1 = e1 + 1.;
    sqrte1p1 = sqrt(e1p1);
    u1bot = sqrte1p1 * cosa12;
    u1 = atan2(tanthk, u1bot);
    u2top = v1 * sinthk + e1p1 * (z2 - z1);
    u2bot = sqrte1p1 * (x2 * cosa12 - y2 * sinthk * sina12);
    u2 = atan2(u2top, u2bot);
    b0 = v1 * sqrt(1. + el * pow(costhk * cosa12, 2.0)) / e1p1;
    du = u2 - u1;
    /* make sure we only compute the short great circle. from GMT4.x */
    if (fabs(du) > M_PI)
    {
        du = (du > 0.0) ? (M_PI * 2.0 - fabs(du)) : (fabs(du) - M_PI * 2.0);
    }
    /* now calculate the distance in KM */
    pdist = b0 * (c2 * (sin(2. * u2) - sin(2. * u1)) + c4 * (sin(4. * u2) - sin(4. * u1)));
    /* output distance in km */
    *distkm = fabs(b0 * c0 * du + pdist);
    /* end of Rudoe's distkm method */
}

/* below is code using Vincenty method */
void distkm_az_baz_Vincenty(double evlo, double evla, double stlo, double stla, double *badgcarc, double *outaz, double *outbaz, double *distkm)
{
    /* wangwt Note:
     * I got these code from GMT5.01 and they got from translating NGS fortran code to C
     * Just a test.
     * Seems if you use those routine in SAC the accuracy is enough.
     * Origional documents in GMT5 is not very clear and I revise it a little bit.
     * wangwt@20130904 */

    /* Translation of NGS FORTRAN code for determination of true distance
    ** and respective forward and back azimuths between two points on the
    ** ellipsoid.  Good for any pair of points that are not antipodal.
    **
    **  INPUT ( GMT note it is radians actually it should be degree)
    **	evla, evlo -- latitude and longitude of event   as first point in degree.
    **	stla, stlo -- latitude and longitude of station as second point in radians.
    **
    **	OUTPUT
    ** 	outaz -- azimuth from first point(Event) to second (Station) in degree clockwise from North.
    **	baz   -- azimuth from second point(Station) back to first point(Event) in degree. This is
    **           actually the back-azimuth.
    **  Code modifid by WangWT from GMT5.01b and then modified from bellow
    ** Modified by P.W. from: http://article.gmane.org/gmane.comp.gis.proj-4.devel/3478
    */

    /* constants for radian-degree conveting. M_PI is defined in math.h */
    double DEG2RAD = M_PI / 180.0;
    double RAD2DEG = 180.0 / M_PI;
    /* earth constant of WGS84. note it is a little different from sac  */
    double EARTHR = 6378.137;             /* Earth Radius,major axis,eg radius at the equator */
    double EARTHFL = 1.0 / 298.257223563; /* Earth Flattening factor,fl=(a-b)/a */
    /* define the max iteration and eps */
    int MAXITER = 50;
    double STOPEPS = 5e-14;
    /* defin*/
    int n_iter = 0;
    /* for az/baz using Vincenty method */
    static double az, c, d, e, r, f, dx, x, y, sa, cx, cy, cz, sx, sy, c2a, cu1, cu2, su1, tu1, tu2, ts, baz, faz;
    /* for distanceKM using Vincenty method */
    static double s;

    /* below is codes from GMT 5.01 */
    f = EARTHFL;
    r = 1.0 - f;
    tu1 = r * tan(evla * DEG2RAD);
    tu2 = r * tan(stla * DEG2RAD);
    cu1 = 1.0 / sqrt(tu1 * tu1 + 1.0);
    su1 = cu1 * tu1;
    cu2 = 1.0 / sqrt(tu2 * tu2 + 1.0);
    ts = cu1 * cu2;
    baz = ts * tu2;
    faz = baz * tu1;
    x = dx = DEG2RAD * (stlo - evlo);
    do
    {
        n_iter++;
        // sincos (x, &sx, &cx);
        sx = sin(x);
        cx = cos(x);
        tu1 = cu2 * sx;
        tu2 = baz - su1 * cu2 * cx;
        sy = sqrt(tu1 * tu1 + tu2 * tu2);
        cy = ts * cx + faz;
        y = atan2(sy, cy);
        sa = ts * sx / sy;
        c2a = -sa * sa + 1.0;
        cz = faz + faz;
        if (c2a > 0.0)
            cz = -cz / c2a + cy;
        e = cz * cz * 2.0 - 1.0;
        c = ((c2a * -3.0 + 4.0) * f + 4.0) * c2a * f / 16.0;
        d = x;
        x = ((e * cy * c + cz) * sy * c + y) * sa;
        x = (1.0 - c) * x * f + dx;
    } while (fabs(d - x) > STOPEPS && n_iter <= MAXITER);
    /* This may happen when they are exactly anti-podal */
    if (n_iter > 50)
    {
        fprintf(stderr, "Near- or actual antipodal points encountered. Precision may be reduced slightly.\n");
        // set s to M_PI for distkm
        s = M_PI;
    }
    // az = (back_az) ? atan2 (tu1, tu2) : atan2 (cu1 * sx, baz * cx - su1 * cu2) + M_PI;
    /* now get the az and baz */
    az = atan2(tu1, tu2);
    baz = atan2(cu1 * sx, baz * cx - su1 * cu2) + M_PI;
    /* now convert to degree and ouput */
    if (*outaz < 0.0 && fabs(*outaz) < 1e-8)
    { // hack code for precision lost
        *outaz = 0.0;
    }
    *outaz = az * RAD2DEG;
    /* make sure azimuth is in [0,360) */
    if (*outaz < 0.0)
        *outaz += 360.0;
    /* do same hack to out baz */
    if (*outbaz < 0.0 && fabs(*outbaz) < 1e-8)
    { // hack code for precision lost
        *outbaz = 0.0;
    }
    /* convert to degree */
    *outbaz = baz * RAD2DEG;
    /* make sure back azimuth is in [0,360) */
    if (*outbaz < 0.0)
        *outbaz += 360.0;
    /* now calculate the distance */
    x = sqrt((1.0 / r / r - 1.0) * c2a + 1.0) + 1.0;
    x = (x - 2.0) / x;
    c = (x * x / 4.0 + 1.0) / (1.0 - x);
    d = (x * 0.375 * x - 1.0) * x;
    s = ((((sy * sy * 4.0 - 3.0) * (1.0 - e - e) * cz * d / 6.0 - e * cy) * d / 4.0 + cz) * sy * d + y) * c * r;
    if (s > M_PI)
    { // again check
        fprintf(stderr, "Near- or actual antipodal points encountered. Precision may be reduced slightly.\n");
        s = M_PI;
    }
    /* now ouput distkm */
    *distkm = s * EARTHR;
    /* seems the gcarc is not very good for this method ? I do not know.
     * add this for same io with sac's distaz routine  wangwt Note*/
    *badgcarc = s * RAD2DEG;
    /* end of Vincenty */
}
