package com.xiaoboluo.d2r;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.view.MotionEvent;
import android.view.View;
import java.io.File;
import java.util.Arrays;

/** 透明悬浮触摸层：与游戏Surface同尺寸，不挤占游戏高度。 | Transparent overlay sharing the game Surface bounds without reducing its height. */
public final class TouchOverlay extends View {
    private static native void nativeEvent(int phase, int slot, float x, float y);
    private static native void nativeResize(float width, float height, float left, float right, float bottom);
    private static native void nativeLayout(float[] out);
    private final float density;
    private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final float[] layout = new float[119];
    private final int[] pointerIds = new int[16];
    private final float[] lastX = new float[16], lastY = new float[16];
    private final int[] rawIds = new int[16];
    private final float[] rawX = new float[16], rawY = new float[16];
    private final String[] gameLabels, menuLabels;
    private final Bitmap joyArt;
    private float safeLeft, safeRight, safeBottom;
    private boolean running, inputEnabled;

    /** labels为本地化表中的15项：12键、摇杆、换组、难度；menuLabels为导航态前五键。 | Pass 15 localized game labels and five navigation labels from the shared string table. */
    public TouchOverlay(Context context, String[] labels, String[] menuLabels) {
        super(context);
        density = getResources().getDisplayMetrics().density;
        Arrays.fill(pointerIds,-1);
        gameLabels = labels == null ? new String[15] : labels.clone();
        this.menuLabels = menuLabels == null ? new String[5] : menuLabels.clone();
        File art = new File(context.getFilesDir(),"dm/overlay/touchart/dm_touch_joy.png");
        joyArt = art.isFile() && art.length() <= 4 * 1024 * 1024 ? BitmapFactory.decodeFile(art.getAbsolutePath()) : null;
        setWillNotDraw(false);
        setFocusable(false);
        setImportantForAccessibility(View.IMPORTANT_FOR_ACCESSIBILITY_NO);
    }
    public void setSafeInsets(int leftPx, int rightPx, int bottomPx) {
        safeLeft=leftPx/density; safeRight=rightPx/density; safeBottom=bottomPx/density;
        resizeNative();
    }
    /** 仅在libdm加载且guest启动成功后开启；暂停时先取消触点。 | Enable only after libdm loads and the guest starts; disabling cancels first. */
    public void setInputEnabled(boolean enabled) {
        if (inputEnabled==enabled) return;
        if (!enabled) cancelTouches();
        inputEnabled=enabled;
        if (enabled) { resizeNative(); postInvalidateOnAnimation(); }
        else invalidate();
    }
    private void resizeNative() {
        if (inputEnabled && getWidth()>0 && getHeight()>0) nativeResize(getWidth()/density,getHeight()/density,safeLeft,safeRight,safeBottom);
    }
    @Override protected void onSizeChanged(int w,int h,int oldw,int oldh) {
        cancelTouches(); resizeNative();
    }
    @Override protected void onAttachedToWindow() {
        super.onAttachedToWindow(); running=true; postInvalidateOnAnimation();
    }
    @Override protected void onDetachedFromWindow() {
        running=false; cancelTouches(); super.onDetachedFromWindow();
    }
    public void cancelTouches() {
        if (!inputEnabled) { Arrays.fill(pointerIds,-1); return; }
        for (int i=0;i<16;++i) if (pointerIds[i]>=0) {
            nativeEvent(3,i,lastX[i],lastY[i]); pointerIds[i]=-1;
        }
        publishRaw();
    }
    private void publishRaw() {
        int count=0;
        if (getWidth()>0 && getHeight()>0) for (int s=0;s<16;++s) if (pointerIds[s]>=0) {
            rawIds[count]=s; rawX[count]=lastX[s]*density/getWidth(); rawY[count]=lastY[s]*density/getHeight(); ++count;
        }
        DmActivity.nativeTouch(count,rawIds,rawX,rawY);
    }
    private int slot(int id) {
        for (int i=0;i<16;++i) if (pointerIds[i]==id) return i;
        return -1;
    }
    private void event(MotionEvent e,int index,int phase,int s) {
        lastX[s]=e.getX(index)/density; lastY[s]=e.getY(index)/density;
        nativeEvent(phase,s,lastX[s],lastY[s]);
    }
    @Override public boolean onTouchEvent(MotionEvent e) {
        if (!inputEnabled) return false;
        int action=e.getActionMasked(), index=e.getActionIndex();
        if (action==MotionEvent.ACTION_DOWN || action==MotionEvent.ACTION_POINTER_DOWN) {
            for (int s=0;s<16;++s) if (pointerIds[s]<0) {
                pointerIds[s]=e.getPointerId(index); event(e,index,0,s); break;
            }
        } else if (action==MotionEvent.ACTION_MOVE) {
            for (int i=0;i<e.getPointerCount();++i) { int s=slot(e.getPointerId(i)); if (s>=0) event(e,i,1,s); }
        } else if (action==MotionEvent.ACTION_UP || action==MotionEvent.ACTION_POINTER_UP) {
            int s=slot(e.getPointerId(index)); if (s>=0) { event(e,index,2,s); pointerIds[s]=-1; }
            if (action==MotionEvent.ACTION_UP) performClick();
        } else if (action==MotionEvent.ACTION_CANCEL) cancelTouches();
        publishRaw(); return true;
    }
    @Override public boolean performClick() { super.performClick(); return true; }
    private String label(int i,boolean navigation) {
        if (i>=15) return Integer.toString(i-14);
        String[] source=navigation && i<5?menuLabels:gameLabels;
        return i<source.length && source[i]!=null?source[i]:"";
    }
    @Override protected void onDraw(Canvas canvas) {
        super.onDraw(canvas); if (!inputEnabled) return; nativeLayout(layout);
        canvas.save(); canvas.scale(density,density);
        boolean navigation=layout[2]!=0;
        for (int i=0;i<23;++i) {
            int p=4+i*5; if (layout[p+3]==0) continue;
            float x=layout[p], y=layout[p+1], r=layout[p+2];
            if (r<=0) continue;
            boolean pressed=layout[p+4]!=0;
            if (i==12 && joyArt!=null) {
                paint.setAlpha(pressed?210:140);
                canvas.drawBitmap(joyArt,null,new RectF(x-r,y-r,x+r,y+r),paint); paint.setAlpha(255);
            } else {
                paint.setStyle(Paint.Style.FILL); paint.setColor(pressed?0x9960a0c0:0x66202024);
                canvas.drawCircle(x,y,r,paint);
                paint.setStyle(Paint.Style.STROKE); paint.setStrokeWidth(1.5f); paint.setColor(0xaadddddd);
                canvas.drawCircle(x,y,r,paint); paint.setStyle(Paint.Style.FILL);
            }
            paint.setColor(Color.WHITE); paint.setTextAlign(Paint.Align.CENTER); paint.setTextSize(Math.min(12,r*0.53f));
            String text=label(i,navigation);
            canvas.drawText(text,x,y-(paint.ascent()+paint.descent())/2,paint);
        }
        canvas.restore(); if (running && isShown()) postInvalidateOnAnimation();
    }
}
