def rk4(f,st,con,h):
    k1 = f(st,con)
    k2 = f(st+(h*(k1/2)),con)
    k3 = f(st+(h*(k2/2)),con)
    k4 = f(st+(h*k3),con)
    y = st + ((h/6)*(k1+(2*k2)+(2*k3)+k4))
    
    return y