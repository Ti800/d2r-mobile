package com.xiaoboluo.d2r;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.LinearGradient;
import android.graphics.RadialGradient;
import android.graphics.Shader;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.view.Choreographer;
import android.view.MotionEvent;
import android.view.View;
import java.io.File;
import java.nio.file.Files;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Locale;
import org.json.JSONObject;
import org.json.JSONArray;

/** 透明悬浮触摸层：与游戏Surface同尺寸，不挤占游戏高度。 | Transparent overlay sharing the game Surface bounds without reducing its height. */
public final class TouchOverlay extends View {
    private static native void nativeEvent(int phase, int slot, float x, float y);
    private static native void nativeResize(float width, float height, float left, float right, float bottom, boolean tablet);
    private static native void nativeLayout(float[] out);
    private final float density;
    private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private static final int GLYPHS=119, HUD=167, DRAW_FLOATS=183;
    private final float[] layout = new float[DRAW_FLOATS];
    // 每个刷新周期比较完整绘图快照；坐标、可见性、按下状态与技能组任一变化才重画。
    // The native layout is checked once per refresh; the view redraws only when it differs from what was last drawn
    // Coordinates, visibility, pressed state and skill set all participate in that snapshot.
    private final float[] probe = new float[DRAW_FLOATS], drawn = new float[DRAW_FLOATS];
    private boolean drawnValid;
    private final Choreographer.FrameCallback poll = new Choreographer.FrameCallback() {
        @Override public void doFrame(long frameTimeNanos) {
            if (!running) return;
            if (inputEnabled && isShown()) { nativeLayout(probe); if (!drawnValid || !Arrays.equals(probe, drawn)) invalidate(); }
            Choreographer.getInstance().postFrameCallback(this);
        }
    };
    private final int[] pointerIds = new int[16];
    private final float[] lastX = new float[16], lastY = new float[16];
    private final int[] rawIds = new int[16];
    private final float[] rawX = new float[16], rawY = new float[16];
    private final String[] gameLabels, menuLabels;
    private final Bitmap joyArt;
    private final File overlayRoot;
    private final HashMap<String,Bitmap> images=new HashMap<>();
    private final Bitmap[] skillAtlases=new Bitmap[9];
    private final String lifeFormat, manaFormat, levelFormat, cancelLabel;
    private JSONObject hudUi;
    private float safeLeft, safeRight, safeBottom;
    private boolean running, inputEnabled;

    /** labels为本地化表中的15项：12键、摇杆、换组、难度；menuLabels为导航态前五键。 | Pass 15 localized game labels and five navigation labels from the shared string table. */
    public TouchOverlay(Context context, String[] labels, String[] menuLabels) {
        super(context);
        density = getResources().getDisplayMetrics().density;
        Arrays.fill(pointerIds,-1);
        gameLabels = labels == null ? new String[15] : labels.clone();
        this.menuLabels = menuLabels == null ? new String[5] : menuLabels.clone();
        lifeFormat=context.getString(R.string.dmhudlife); manaFormat=context.getString(R.string.dmhudmana);
        levelFormat=context.getString(R.string.dmandroidhudlevel); cancelLabel=context.getString(R.string.dmtccancel);
        overlayRoot=new File(context.getFilesDir(),"dm/overlay");
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
        drawnValid=false; invalidate();
        if (enabled) {
            images.clear();
            for(int i=0;i<skillAtlases.length;i++) skillAtlases[i]=image("touchart/dm_native_skill_"+i+".png");
            try { hudUi=new JSONObject(new String(Files.readAllBytes(new File(overlayRoot,"hud_v4_ui/ui.json").toPath()),StandardCharsets.UTF_8)); }
            catch(Exception ignored) { hudUi=null; }
            resizeNative();
        }
    }
    private void resizeNative() {
        if (inputEnabled && getWidth()>0 && getHeight()>0) nativeResize(getWidth()/density,getHeight()/density,safeLeft,safeRight,safeBottom,
                getResources().getConfiguration().smallestScreenWidthDp>=600);
    }
    @Override protected void onSizeChanged(int w,int h,int oldw,int oldh) {
        cancelTouches(); resizeNative();
    }
    @Override protected void onAttachedToWindow() {
        super.onAttachedToWindow(); running=true; drawnValid=false; invalidate(); Choreographer.getInstance().postFrameCallback(poll);
    }
    @Override protected void onDetachedFromWindow() {
        running=false; Choreographer.getInstance().removeFrameCallback(poll); cancelTouches(); super.onDetachedFromWindow();
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
        String text=i<source.length && source[i]!=null?source[i]:"";
        // 换组字段同时决定文字，避免快照变了却仍画相同的组别。 | The skill-set field also changes the label so a changed snapshot displays the new group.
        if(i==13) return layout[3]!=0?"Ⅱ":"Ⅰ";
        if(i==11 && layout[HUD+15]!=0) return cancelLabel;
        return text;
    }
    // 图像只从用户导入的已校验资源读取，不放进安装包。
    // Images are read only from the user's validated imported resources, never from the APK.
    private Bitmap image(String name) {
        if(images.containsKey(name)) return images.get(name);
        File file=new File(overlayRoot,name);
        Bitmap bitmap=file.isFile() && file.length()<=8*1024*1024 ? BitmapFactory.decodeFile(file.getAbsolutePath()):null;
        images.put(name,bitmap); return bitmap;
    }
    private void socket(Canvas c,float x,float y,float r,boolean held,boolean empty,boolean primary) {
        float side=2*r;
        paint.setStyle(Paint.Style.FILL);
        paint.setShader(new LinearGradient(x,y-r,x,y+r,new int[]{empty?0x38424747:0xe6424747,empty?0x380e1012:0xf20e1012,empty?0x29291c1a:0xb3291c1a},null,Shader.TileMode.CLAMP));
        c.drawCircle(x,y,r-side*.018f,paint); paint.setShader(null);
        paint.setColor(empty?0x290d0d0d:0xf00d0d0d); c.drawCircle(x,y,r-side*.055f,paint);
        paint.setStyle(Paint.Style.STROKE); paint.setStrokeWidth(side*(held?.018f:empty?.012f:.009f));
        paint.setColor(empty?(held?0xccb8b8b8:0x999e9e9e):(held?0xb8a3a3a3:0x8c424a47));
        c.drawCircle(x,y,r-side*.055f,paint);
        if(primary && !empty) { paint.setColor(0x854f3b38);paint.setStrokeWidth(side*.012f);c.drawCircle(x,y,r-side*.035f,paint); }
        if(empty) {
            paint.setColor(held?0xbf99a6a3:0x4099a6a3);paint.setStrokeWidth(side*.018f);
            float arm=side*.055f;c.drawLine(x-arm,y,x+arm,y,paint);c.drawLine(x,y-arm,x,y+arm,paint);
        }
        paint.setStyle(Paint.Style.FILL);
    }
    private void text(Canvas c,String value,float x,float y,float size) {
        paint.setColor(0xffcdb98c);paint.setTextSize(size);paint.setTextAlign(Paint.Align.CENTER);
        paint.setShadowLayer(1.2f,0,1,Color.BLACK);c.drawText(value,x,y,paint);paint.clearShadowLayer();
    }
    private void skill(Canvas c,int i,float x,float y,float r,boolean held) {
        int p=GLYPHS+i*8, atlas=(int)layout[p+1];
        Bitmap bitmap=atlas>=0 && atlas<skillAtlases.length?skillAtlases[atlas]:null;
        int sx=(int)layout[p+2],sy=(int)layout[p+3],sw=(int)layout[p+4],sh=(int)layout[p+5];
        boolean valid=layout[p]==2 && bitmap!=null && sw>0 && sh>0 && sx>=0 && sy>=0 && sx+sw<=bitmap.getWidth() && sy+sh<=bitmap.getHeight();
        socket(c,x,y,r,held,!valid,i==0);
        if(!valid) return;
        c.save();Path clip=new Path();clip.addCircle(x,y,r*.82f,Path.Direction.CW);c.clipPath(clip);
        boolean spent=((int)layout[p+6]&2)!=0 && layout[p+7]<=0;
        if(spent) { ColorMatrix m=new ColorMatrix();m.setSaturation(0);paint.setColorFilter(new ColorMatrixColorFilter(m)); }
        paint.setAlpha(spent?140:held?255:240);
        float fit=r*1.52f/Math.max(sw,sh);
        c.drawBitmap(bitmap,new Rect(sx,sy,sx+sw,sy+sh),new RectF(x-sw*fit/2,y-sh*fit/2,x+sw*fit/2,y+sh*fit/2),paint);
        paint.setAlpha(255);paint.setColorFilter(null);c.restore();
        if(((int)layout[p+6]&1)!=0) text(c,"Ⅱ",x+r*.70f,y-r*.50f,r*.40f);
        if(((int)layout[p+6]&2)!=0) text(c,Integer.toString(Math.max(0,(int)layout[p+7])),x+r*.65f,y+r*.65f,r*.42f);
    }
    private float hudX(double x,float u) { return getWidth()/density/2+((float)x-1434)*u; }
    private void hudImage(Canvas c,String name,double x,double y,float u) {
        Bitmap b=image("hud_v4_ui/"+name); if(b==null)return;
        float left=hudX(x,u),top=(float)y*u;
        paint.setColor(Color.WHITE);c.drawBitmap(b,null,new RectF(left,top,left+b.getWidth()*u,top+b.getHeight()*u),paint);
    }
    // 字形位置、底图和数值基线来自与iOS相同的生成器，游戏层仍负责药水与球液。
    // Glyph placement, base art and baselines come from the iOS generator; the game still owns potions and orb liquid.
    private void hudText(Canvas c,String value,JSONObject glyphs,double x,double baseline,boolean right,float u) throws Exception {
        double advance=0;
        for(int i=0;i<value.length();i++) { JSONObject g=glyphs.optJSONObject(value.substring(i,i+1)); if(g!=null)advance+=g.optDouble("adv"); }
        double pen=x-(right?advance:advance/2);
        for(int i=0;i<value.length();i++) {
            JSONObject g=glyphs.optJSONObject(value.substring(i,i+1));if(g==null)continue;
            if(!g.isNull("file")) hudImage(c,g.getString("file"),pen+g.optDouble("dx"),baseline+g.optDouble("dy"),u);
            pen+=g.optDouble("adv");
        }
    }
    private void hud(Canvas c) {
        if(hudUi==null || layout[HUD]==0) return;
        float u=getHeight()/density/1320;
        try {
            JSONObject base=hudUi.getJSONObject("base");hudImage(c,"base_full.png",base.getDouble("x"),base.getDouble("y"),u);
            JSONArray slots=hudUi.getJSONArray("slots"), inner=hudUi.getJSONArray("inner"), menu=slots.getJSONArray(5);
            boolean alert=layout[HUD+12]>0 || layout[HUD+13]>0, held=layout[4+10*5+4]!=0;
            Bitmap button=image("hud_v4_ui/"+(alert?"lvl_":"lvl_d")+(held?1:0)+".png");
            if(button!=null) c.drawBitmap(button,null,new RectF(hudX(menu.getDouble(0),u),(float)menu.getDouble(1)*u,hudX(menu.getDouble(2),u),(float)menu.getDouble(3)*u),paint);
            JSONObject e=hudUi.getJSONObject("exp");
            float x=hudX(e.getDouble("x0"),u),y=(float)e.getDouble("y")*u,w=(float)(e.getDouble("x1")-e.getDouble("x0"))*u,h=(float)e.getDouble("h")*u;
            paint.setColor(0xff16120e);c.drawRect(x,y,x+w,y+h,paint);
            paint.setColor(0xffc4a046);c.drawRect(x+u,y+u,x+u+(w-2*u)*Math.max(0,Math.min(1000,layout[HUD+2]))/1000,y+h-u,paint);
            paint.setColor(0xff463a28);paint.setStyle(Paint.Style.STROKE);paint.setStrokeWidth(u);c.drawRect(x,y,x+w,y+h,paint);paint.setStyle(Paint.Style.FILL);
            paint.setColor(0xff16120e);for(int i=1;i<20;i++)c.drawRect(x+w*i/20,y+u,x+w*i/20+u,y+h-u,paint);
            JSONObject lv=hudUi.getJSONObject("lv"), vital=hudUi.getJSONObject("vital"),badge=hudUi.getJSONObject("badge");
            hudText(c,String.format(Locale.ROOT,levelFormat,(int)layout[HUD+1],(int)layout[HUD+2]/10),lv.getJSONObject("glyphs"),lv.getDouble("x"),lv.getDouble("baseline"),false,u);
            hudText(c,String.format(Locale.ROOT,lifeFormat,(int)layout[HUD+8],(int)layout[HUD+9]),vital.getJSONObject("glyphs"),vital.getDouble("left_x"),vital.getDouble("baseline"),false,u);
            hudText(c,String.format(Locale.ROOT,manaFormat,(int)layout[HUD+10],(int)layout[HUD+11]),vital.getJSONObject("glyphs"),vital.getDouble("right_x"),vital.getDouble("baseline"),false,u);
            for(int i=1;i<=4;i++) if(layout[HUD+2+i]>0) {
                JSONArray s=slots.getJSONArray(i);
                hudText(c,Integer.toString((int)layout[HUD+2+i]),badge.getJSONObject("glyphs"),s.getDouble(0)+inner.getDouble(2)+badge.getDouble("dx"),s.getDouble(1)+inner.getDouble(3)+badge.getDouble("dy"),true,u);
            }
            int players=(int)layout[HUD+7];
            if(players>=1 && players<=8) {
                JSONArray s=slots.getJSONArray(0);Bitmap b=image("hud_v4_ui/diff_"+players+".png");
                if(b!=null) hudImage(c,"diff_"+players+".png",s.getDouble(0)+(inner.getDouble(0)+inner.getDouble(2)+1-b.getWidth())/2,s.getDouble(1)+(inner.getDouble(1)+inner.getDouble(3)+1-b.getHeight())/2,u);
            }
        } catch(Exception ignored) { /* 缺损美术不制造HUD数据。 | Missing art never fabricates HUD values. */ }
    }
    @Override protected void onDraw(Canvas canvas) {
        super.onDraw(canvas); if (!inputEnabled) return; nativeLayout(layout); System.arraycopy(layout, 0, drawn, 0, layout.length); drawnValid=true;
        canvas.save(); canvas.scale(density,density);
        boolean navigation=layout[2]!=0;
        for (int i=0;i<23;++i) {
            int p=4+i*5; if (layout[p+3]==0) continue;
            float x=layout[p], y=layout[p+1], r=layout[p+2];
            if (r<=0) continue;
            boolean pressed=layout[p+4]!=0;
            if(!navigation && i<=5) { skill(canvas,i,x,y,r,pressed); continue; }
            if(!navigation && i>=6 && i<=10 && hudUi!=null) continue;
            if(i==12 && layout[HUD+14]!=0) continue;
            if(!navigation && i==13) { socket(canvas,x,y,r,pressed,false,false);text(canvas,label(i,false),x,y+r*.18f,r*.55f);continue; }
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
        hud(canvas);
        canvas.restore();
    }
}
