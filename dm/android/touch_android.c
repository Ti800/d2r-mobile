#include "touch_android.h"
#include "../ios/Sources/dm_touch_controls.h"
#include "../host/dm_input.h"
#include "../host/dm_cursor.h"
#include "../host/dm_actions.h"
#include "../host/dm_hud.h"
#include <pthread.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

uint64_t now_ns(void);
void dm_uiscreen_set_panel(int panel);
void dm_tp_request(void);
/* 绑定观测桥未接入时不捏造技能绑定；基础键位仍走共用状态机。 | Do not fabricate skill bindings if the observation bridge is absent; basic keys still use the shared state machine. */
void dm_touch_game_set_scene(int game, uint64_t scene, uint64_t world) __attribute__((weak));

typedef struct { int phase, slot; float x, y; double time; } touch_event;
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static dm_tc_state tc;
static dm_tc_scene scene;
static touch_event queue[256];
static unsigned head, tail;
static int ready, focused = 1, cancel, resizing, ui_drag;
static float width = 800, height = 375, safe_left, safe_right, safe_bottom;
static uint64_t epoch = 1, world = 1, pulse_epoch, pulse_sample, resume_sample;
static double pulse_hold_start;
static unsigned seen_world, seen_mouse, seen_tab, seen_use, seen_ident, seen_long, seen_item, seen_stack;
/* 前端页面（主菜单 / 选角 / 建角 / 读取画面，即不在局内），同 iOS front_ios.m frontendMode：不画控件，第一根手指直接当原生点击交给游戏界面
   （dm_ui_touch → 鼠标消息，dm_cursor.c），纵向拖过 10 点改成滚动。只在装了触屏覆盖层（DM_ROMFS_OVERLAY）时这样：覆盖层把选角页换成键鼠版
   （「开始」「创建新角色」靠点）；没有覆盖层时前端是原版手柄页，仍走导航键。
   Front-end pages (main menu / character select / creation / loading, i.e. not in a run), as iOS front_ios.m frontendMode: no controls are drawn, and
   the first finger goes straight to the game UI as a native click (dm_ui_touch → mouse messages, dm_cursor.c); a vertical drag past 10 points scrolls.
   Only with the touch overlay (DM_ROMFS_OVERLAY): it turns character select into the keyboard-and-mouse page ("Play", "Create New" are tapped);
   without it the front end is the original controller page and keeps the navigation keys. */
static int frontend, fe_slot = -1, fe_drag;
static float fe_x0, fe_y0;
static int touch_overlay(void) { static int v = -1; if (v < 0) { const char *o = getenv("DM_ROMFS_OVERLAY"); v = o && *o; } return v; }
static void fe_end_locked(void) {
    if (fe_slot >= 0) { if (fe_drag) dm_ui_drag_end(); else dm_ui_touch(3, 0, 0); }
    fe_slot = -1; fe_drag = 0;
}
static void fe_event_locked(const touch_event *e) {
    if (e->phase == 0 && fe_slot < 0) { fe_slot = e->slot; fe_x0 = e->x; fe_y0 = e->y; fe_drag = 0; }
    if (e->slot != fe_slot) return;
    float nx = e->x / width, ny = e->y / height, dy = fabsf(e->y - fe_y0);
    if (e->phase == 1 && !fe_drag && dy > 10 && dy > fabsf(e->x - fe_x0) && dm_ui_fe_drag_ok()) {
        fe_drag = 1; dm_ui_touch(3, 0, 0); dm_ui_drag_begin(1, fe_x0 / width, fe_y0 / height, 0);
    }
    if (fe_drag) { if (e->phase == 1) dm_ui_drag_move(nx, ny); else fe_end_locked(); return; }
    if (e->phase == 3) { fe_end_locked(); return; }
    dm_ui_touch(e->phase, nx, ny);
    if (e->phase == 2) fe_slot = -1;
}

static void init_locked(void) {
    if (ready) return;
    dm_tc_set_tall_scale(1);
    dm_tc_init(&tc, width, height);
    tc.mouse_click_mode = 1;
    dm_tc_set_navigation_mode(&tc, 1);
    dm_tc_enable_targets(&tc, dm_tc_nav_mask(1));
    ready = 1;
}
static void cancel_locked(void) {
    dm_tc_cancel_all(&tc);
    if (ui_drag) dm_ui_drag_end();
    ui_drag = 0;
    fe_end_locked();
    pulse_epoch = pulse_sample = resume_sample = 0;
    pulse_hold_start = 0;
    dm_cursor_clear();
}
void dm_android_touch_resize(float w, float h, float l, float r, float b) {
    if (!isfinite(w) || !isfinite(h) || w <= 0 || h <= 0) return;
    pthread_mutex_lock(&lock);
    width = w; height = h; safe_left = fmaxf(0,l); safe_right = fmaxf(0,r); safe_bottom = fmaxf(0,b);
    resizing = cancel = 1; head = tail = 0;
    pthread_mutex_unlock(&lock);
}
void dm_android_touch_focus(int active) {
    pthread_mutex_lock(&lock);
    focused = !!active; cancel = 1; head = tail = 0;
    pthread_mutex_unlock(&lock);
}
void dm_android_touch_event(int phase, int slot, float x, float y) {
    if (phase < 0 || phase > 3 || slot < 0 || slot >= DM_TC_MAX_TOUCH || !isfinite(x) || !isfinite(y)) return;
    pthread_mutex_lock(&lock);
    if (focused) {
        /* 溢出只取消，绝不遗留按住的摇杆/按键。 | Overflow cancels instead of leaving a stick/key held. */
        if (tail - head == 256) { cancel = 1; head = tail = 0; }
        else { queue[tail++ % 256] = (touch_event){phase, slot, x, y, now_ns() / 1e9}; }
    }
    pthread_mutex_unlock(&lock);
}
void dm_android_touch_uimsg(const char *cat, const char *act, const char *arg) {
    if (!cat || !act) return;
    pthread_mutex_lock(&lock);
    init_locked();
    dm_tc_scene before = scene;
    dm_tc_set_panel_state(dm_ui_panel_state());
    int nav = dm_tc_navigation(&scene, cat, act, arg ? arg : "");
    if (before.game != scene.game || before.panels != scene.panels) {
        ++epoch; if (before.game != scene.game) ++world;
        cancel_locked(); head = tail = 0;
    }
    int fe = !scene.game && touch_overlay();
    if (fe != frontend) { cancel_locked(); head = tail = 0; frontend = fe; }
    dm_tc_set_navigation_mode(&tc, nav);
    dm_tc_set_joy_block(&tc, !!(dm_ui_panel_state() & DM_PS_TEXT));
    dm_tc_enable_targets(&tc, nav ? dm_tc_nav_mask(!dm_tc_scene_dead()) : ((1u << DM_TC_COUNT) - 1));
    dm_uiscreen_set_panel(nav && !dm_tc_scene_dead() && !dm_tc_scene_dialog_only(&scene));
    if (dm_touch_game_set_scene) dm_touch_game_set_scene(!nav, epoch, world);
    pthread_mutex_unlock(&lock);
}
static void sync_host(void) {
    dm_tc_set_panel_state(dm_ui_panel_state());
    dm_tc_set_tap_box(dm_ui_tap_box());
    dm_tc_set_cur_stack(dm_ui_cur_stack_id());
    dm_tc_set_hover_unid(dm_ui_hover_unid());
    dm_hud_info info; dm_hud_get(&info);
    dm_tc_set_holding(&tc, info.valid && info.holding && !dm_ui_targeting());
    tc.belt_rows = dm_belt_rows(&tc.belt_open);
}
static void requests(void) {
    if (seen_tab != tc.tab_tap_seq) { seen_tab = tc.tab_tap_seq; dm_ui_tab_go(); }
    if (seen_use != tc.use_seq) { seen_use = tc.use_seq; dm_ui_use_item(); }
    if (seen_ident != tc.ident_seq) { seen_ident = tc.ident_seq; dm_ui_identify_hovered(); }
    if (seen_long != tc.long_press_seq) { seen_long = tc.long_press_seq; dm_ui_long_press(); }
    if (seen_item != tc.item_tap_seq) { seen_item = tc.item_tap_seq; dm_ui_item_tap(); }
    if (seen_stack != tc.stack_take_seq) { seen_stack = tc.stack_take_seq; dm_stash_take(1); }
    if (seen_world != tc.world_tap_seq) {
        seen_world = tc.world_tap_seq;
        float nx = tc.world_tap_x / width, ny = tc.world_tap_y / height;
        dm_ui_world_tap(nx, ny, nx, ny);
    }
    if (seen_mouse != tc.mouse_click_seq) {
        seen_mouse = tc.mouse_click_seq;
        float nx = tc.mouse_click_x / width, ny = tc.mouse_click_y / height;
        dm_ui_touch(0,nx,ny); dm_ui_touch(2,nx,ny);
    }
    if (tc.diff_request) { dm_hud_request_players(tc.diff_request); tc.diff_request = 0; }
    if (tc.belt_request) {
        int br = tc.belt_request; tc.belt_request = 0;
        if (br == 1 || br == 2) dm_belt_expand(br == 1);
        else dm_belt_use(br & 15, (br >> 4) & 15);
    }
}
static void release_pointer(dm_tc_finger *f) {
    float lift = DM_TC_POINTER_LIFT_PT * dm_tc_scale_for_size(width,height);
    int kind = dm_ui_drop_at_now(f->x/width, (f->y-lift)/height, f->y/height);
    float x,y;
    if (f->shop_drag) {
        if (kind == 11) dm_ui_stack_cube_at(f->aim_origin_x/width,f->aim_origin_y/height);
        else if (kind == 5 && dm_ui_drop_bag()) {
            if (tc.pointer_stack) dm_ui_stack_take_at(f->aim_origin_x/width,f->aim_origin_y/height);
            else dm_ui_shop_buy_at(f->aim_origin_x/width,f->aim_origin_y/height);
        }
        tc.pointer_x = f->aim_origin_x; tc.pointer_y = f->aim_origin_y;
    } else if ((kind == 5 && dm_ui_drop_point(&x,&y)) || (kind == 0 && dm_ui_origin_point(&x,&y))) dm_tc_pointer_release_at(&tc,x*width,y*height);
    else { if (kind == 7) { dm_shop_sell_cursor(); kind = 0; } dm_tc_pointer_release(&tc,kind); }
}
void dm_android_touch_step(uint64_t guest_now, uint64_t *buttons, int32_t *lx, int32_t *ly, int32_t *rx, int32_t *ry) {
    (void)guest_now;
    pthread_mutex_lock(&lock);
    init_locked();
    if (cancel) { cancel_locked(); cancel = 0; }
    if (resizing) {
        dm_tc_set_safe_insets(safe_left,safe_right); dm_tc_set_edge_bottom(safe_bottom);
        dm_tc_resize(&tc,width,height); resizing = 0;
    }
    if (!focused) { pthread_mutex_unlock(&lock); return; }
    sync_host();
    /* 本次取手柄前排队的触点整批在这里套用：按下与抬起落在同一批（点得比游戏两次取手柄的间隔还短、或游戏卡住没取输入）时，状态机算出的键在批末已经松开，
       游戏一次也看不到——iOS 由前端异步调 dm_input_set，按下沿进 hle_hid.c 的 g_press_latch 锁存，这里同样：批内任何时刻按着过的键这次一并给出，下一次照实际值。
       The touches queued before this controller read are applied here as one batch: when press and release land in the same batch (a tap shorter than the
       interval between two reads, or the game stalling its input reads), the state machine's key is already up at the end of the batch and the game never
       sees it. On iOS the front end calls dm_input_set asynchronously and the press edge is latched in hle_hid.c g_press_latch; likewise here, keys held at
       any point in the batch are delivered with this read, and the next read reports the actual state. */
    uint64_t pressed = 0;
    while (head != tail) {
        touch_event e = queue[head++ % 256];
        if (frontend) { fe_event_locked(&e); continue; }
        dm_tc_finger *f = &tc.finger[e.slot];
        if (e.phase == 0) {
            if (dm_tc_belt_outside(&tc,e.x,e.y)) { dm_belt_expand(0); continue; }
            dm_tc_begin_at(&tc,e.slot,e.x,e.y,e.time);
            if (!tc.navigation_mode && f->target == DM_TC_PORTAL) { dm_tc_cancel_finger(&tc,e.slot); dm_tp_request(); }
            if (tc.navigation_mode && tc.pointer_slot == e.slot) dm_tc_pointer_item(&tc,dm_ui_item_at_now(e.x/width,e.y/height));
        } else if (e.phase == 1 || e.phase == 2) {
            dm_tc_move_at(&tc,e.slot,e.x,e.y,e.time);
            if (tc.navigation_mode && tc.pointer_slot == e.slot && f->ui_drag) {
                if (!ui_drag) {
                    ui_drag = 1; dm_ui_drag_lift(DM_TC_POINTER_LIFT_PT*dm_tc_scale_for_size(width,height)/height);
                    dm_ui_drag_begin(f->ui_drag,f->aim_origin_x/width,f->aim_origin_y/height,44/height);
                }
                dm_ui_drag_move(e.x/width,e.y/height);
            }
            if (e.phase == 2) {
                tc.tap_button=0;
                if (tc.navigation_mode && dm_ui_bind_active()==2) {
                    float nx,ny;
                    if (dm_ui_bind_skill_hit(&nx,&ny)==1) { tc.tap_button=DM_BTN_Y; tc.tap_x=nx*width; tc.tap_y=ny*height; }
                }
                if (tc.navigation_mode && tc.pointer_slot == e.slot && (f->pointer_dragged || f->shop_drag)) release_pointer(f);
                if (ui_drag && tc.pointer_slot == e.slot) { dm_ui_drag_end(); ui_drag = 0; }
                dm_tc_end_at(&tc,e.slot,e.time);
            }
        } else {
            /* 系统取消物品拖动时回原位，不当作丢弃/购买。 | System cancellation returns dragged items; it never buys or discards. */
            if (tc.pointer_slot == e.slot && f->pointer_dragged && !f->shop_drag) {
                dm_tc_pointer_release(&tc,0); dm_tc_end_at(&tc,e.slot,e.time);
            } else dm_tc_cancel_finger(&tc,e.slot);
            if (ui_drag) { dm_ui_drag_end(); ui_drag = 0; }
        }
        pressed |= tc.buttons;
        requests();
    }
    double now = now_ns()/1e9;
    /* 技能长按重绑复用宿主流程；不新增一套技能选择逻辑。 | A skill long hold reuses the host rebind flow rather than duplicating skill selection. */
    if (!tc.navigation_mode) for (int i=0;i<DM_TC_MAX_TOUCH;++i) {
        dm_tc_finger *f = &tc.finger[i];
        if (f->active && f->target>=DM_TC_SKILL1 && f->target<=DM_TC_SKILL4 && now-f->start_time_s>=0.5 && hypotf(f->x-f->aim_origin_x,f->y-f->aim_origin_y)<=10*dm_tc_scale_for_size(width,height)) {
            uint64_t key=tc.layout.target[f->target].button|(tc.skill_set?DM_BTN_ZL:0);
            dm_tc_cancel_finger(&tc,i); dm_ui_bind_begin(key);
        }
    }
    dm_tc_tick(&tc,now); requests();
    uint64_t sample = dm_input_samples();
    if (tc.pulse_button && pulse_epoch != tc.pulse_epoch) {
        pulse_epoch = tc.pulse_epoch; pulse_sample = sample; pulse_hold_start = 0;
    }
    if (tc.pulse_button && !dm_cursor_pending()) {
        if (!pulse_hold_start) pulse_hold_start = now;
        if (sample - pulse_sample >= 4 && now - pulse_hold_start >= tc.pulse_hold_s) {
            dm_tc_pulse_end(&tc,pulse_epoch); resume_sample = sample + 2;
        }
    }
    if (!tc.pulse_button && dm_tc_pulse_pending(&tc) && resume_sample && sample >= resume_sample) { dm_tc_pulse_resume(&tc); resume_sample = 0; }
    if (tc.pointer_valid) dm_cursor_set_screen(tc.pointer_x,tc.pointer_y,width,height); else dm_cursor_clear();
    int active = 0; for (int i=0;i<DM_TC_MAX_TOUCH;++i) active |= tc.finger[i].active;
    dm_ui_hand_idle(!active && !tc.pulse_button && !dm_tc_pulse_pending(&tc) && !dm_cursor_pending() && !dm_ui_touch_active());
    *buttons |= tc.buttons | pressed;
    if (!*lx && !*ly) { *lx = tc.lx; *ly = tc.ly; }
    if (!*rx && !*ry) { *rx = tc.rx; *ry = tc.ry; }
    pthread_mutex_unlock(&lock);
}
void dm_android_touch_layout(float out[DM_ANDROID_TOUCH_DRAW_FLOATS]) {
    pthread_mutex_lock(&lock); init_locked();
    memset(out,0,sizeof(float)*DM_ANDROID_TOUCH_DRAW_FLOATS);
    out[0]=tc.layout.width; out[1]=tc.layout.height; out[2]=tc.navigation_mode; out[3]=tc.skill_set;
    for (int i=0;i<23;++i) {
        float *p = out+4+i*5;
        dm_tc_target t = i<DM_TC_COUNT ? tc.layout.target[i] : i==13 ? tc.layout.swap : i>=15 ? tc.layout.pick[i-15] : tc.layout.diff;
        if (i==12) {
            p[0]=tc.joy_slot>=0?tc.joy_origin_x:tc.layout.joy_x; p[1]=tc.joy_slot>=0?tc.joy_origin_y:tc.layout.joy_y;
            p[2]=tc.layout.joy_radius; p[3]=!tc.navigation_mode; p[4]=tc.joy_slot>=0;
        } else {
            p[0]=t.x; p[1]=t.y; p[2]=t.r;
            p[3]=i<DM_TC_COUNT?!!(tc.enabled_targets&(1u<<i)):!tc.navigation_mode && (i<15 || tc.diff_open);
            for (int j=0;j<DM_TC_MAX_TOUCH;++j) if (tc.finger[j].active && tc.finger[j].target==(i==13?DM_TC_SWAP:i==14?DM_TC_DIFF:i)) p[4]=1;
        }
        if (frontend) p[3]=0;   /* 前端页面不画控件（见 frontend） | No controls drawn on front-end pages (see frontend) */
    }
    pthread_mutex_unlock(&lock);
}
