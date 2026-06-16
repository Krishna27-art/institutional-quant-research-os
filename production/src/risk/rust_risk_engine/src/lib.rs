use std::os::raw::c_char;
use std::ffi::CStr;

#[repr(C)]
pub struct RiskLimits {
    pub max_position_value: f64,
    pub max_daily_orders: i32,
    pub max_daily_trades: i32,
    pub max_exposure: f64,
    pub max_leverage: f64,
}

#[repr(C)]
pub struct OrderContext {
    pub current_position_value: f64,
    pub current_exposure: f64,
    pub daily_orders_count: i32,
    pub daily_trades_count: i32,
    pub order_price: f64,
    pub order_quantity: i32,
}

#[no_mangle]
pub extern "C" fn check_order_limits(
    limits: *const RiskLimits,
    ctx: *const OrderContext,
    _symbol: *const c_char
) -> bool {
    if limits.is_null() || ctx.is_null() {
        return false;
    }
    
    let l = unsafe { &*limits };
    let c = unsafe { &*ctx };
    
    let order_value = c.order_price * (c.order_quantity as f64).abs();
    
    if c.daily_orders_count >= l.max_daily_orders {
        return false;
    }
    if c.current_position_value + order_value > l.max_position_value {
        return false;
    }
    if c.current_exposure + order_value > l.max_exposure {
        return false;
    }
    
    true
}

#[no_mangle]
pub extern "C" fn calculate_margin(price: f64, quantity: i32, margin_rate: f64) -> f64 {
    price * (quantity as f64).abs() * margin_rate
}
