# OAB-1CMT-FOD-LIN-WTD-IEXP-OADD.R

mod <- function() {
  ini({
    talag <- log(1.1)
    tka <- log(1.5)
    tcl <- log(1.5)
    tv <- log(3)
    eta.ka ~ 1
    eta.cl ~ 1
    eta.v ~ 1
    add.err <- 0.1
  })
  
  model({
    lagD <- exp(talag)
    ka <- exp(tka + eta.ka)
    cl <- exp(tcl + eta.cl)
    v <- exp(tv + eta.v)
    d/dt(depot) = -ka * depot
    alag(depot) <- lagD
    d/dt(center) = ka * depot - cl / v * center
    cp = center / v
    cp ~ add(add.err)
  })
}