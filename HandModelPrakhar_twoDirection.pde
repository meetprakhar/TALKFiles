float a1;
float scaleLevel = 1.0,x=180; // scale 



void setup() {
  size(500, 900, P3D);
  background(20);
} // func 
 
void draw() {
  background(0);
  lights();
  translate(width/2, height*0.75);
  rotateY(a1);
  rotateX(PI/2);

  box(20*4+5*3*1.5,200+50, 12);
  translate(-(20+5)*1.5, -(250/2),0);
  finger(180,20,10,x,90,90,rotateXToUse);
  translate((20+5), 0,0);
  finger(200,20,10,x,180,180,rotateXToUse);
  translate((20+5), 0,0);
  finger(190,20,10,x,90,90,rotateXToUse);
  translate((20+5), 0,0);
  finger(150,20,10,x,90,90,rotateXToUse);
  translate(-(20+5)*4,0.75*250,0);
  thumb(170,20,10,10,160,90,-90);
  
  
  a1+=.022;
} // func 

  
void finger(float h,float w,float thickness,float t_oa,float t_ab,float t_bc,float t_z){
  //(PI/180)*theta = theta_rad
  float r = 0.75*w*0.5;
  float a = 0.46*h,b=0.35*h,c=0.2*h;
  PShape s1;
  PShape a1,a2,a3;
  
  
  s1 = createShape(SPHERE,r);
  s1.setStroke(color(255,150,0));
  s1.setStrokeWeight(1);
  s1.setFill(color(255,150,0));
  
  a1 = createShape(BOX,w,a,thickness);
  a1.setStroke(color(255,150,0));
  a1.setStrokeWeight(1);
  a1.setFill(color(255,150,0));
  
  a2 = createShape(BOX,w,b,thickness);
  a2.setStroke(color(255,150,0));
  a2.setStrokeWeight(1);
  a2.setFill(color(255,150,0));
  
  a3 = createShape(BOX,w,c,thickness);
  a3.setStroke(color(255,150,0));
  a3.setStrokeWeight(1);
  a3.setFill(color(255,150,0));
  
  float X,Y,Z=0;
  
  pushMatrix();
  shape(s1);
  
  a1.rotate(radians(t_oa),1,0,0);
  Z = (a*0.5+r)*cos(radians(t_oa-90));
  Y = (a*0.5+r)*sin(radians(t_oa-90));
  a1.rotate(radians(t_z),0,0,1);
  X = Y*cos(radians(t_z-90));
  Y = Y*sin(radians(t_z-90));
  translate(X,Y,Z);
  shape(a1);
  
  Z = (0.5*a+r)*cos(radians(t_oa-90));
  Y = (0.5*a+r)*sin(radians(t_oa-90));
  X = Y*cos(radians(t_z-90));
  Y = Y*sin(radians(t_z-90));
  translate(X,Y,Z);
  shape(s1);
  
  t_ab = -180+(t_oa+t_ab);
  a2.rotate(radians(t_ab),1,0,0);
  Z = (b*0.5+r)*cos(radians(t_ab-90));
  Y = (b*0.5+r)*sin(radians(t_ab-90));
  a2.rotate(radians(t_z),0,0,1);
  X = Y*cos(radians(t_z-90));
  Y = Y*sin(radians(t_z-90));
  translate(X,Y,Z);
  
  shape(a2);
  Z = (0.5*b+r)*cos(radians(t_ab-90));
  Y = (0.5*b+r)*sin(radians(t_ab-90));
  X = Y*cos(radians(t_z-90));
  Y = Y*sin(radians(t_z-90));
  translate(X,Y,Z);
  shape(s1);
  
  t_bc = -180+(t_ab+t_bc);
  a3.rotate(radians(t_bc),1,0,0);
  Z = (c*0.5+r)*cos(radians(t_bc-90));
  Y = (c*0.5+r)*sin(radians(t_bc-90));
  a3.rotate(radians(t_z),0,0,1);
  X = Y*cos(radians(t_z-90));
  Y = Y*sin(radians(t_z-90));
  translate(X,Y,Z);
  shape(a3);
  
  popMatrix();
}

void thumb(float h,float w,float thickness,float t_oa,float t_ab,float t_bc,float t_z){
  //(PI/180)*theta = theta_rad
  float r = 0.75*w*0.5;
  float a = 0.46*h,b=0.35*h,c=0.2*h;
  PShape s1,s2;
  PShape a1,a2;
  
  
  s1 = createShape(SPHERE,r);
  s1.setStroke(color(255,150,0));
  s1.setStrokeWeight(1);
  s1.setFill(color(255,150,0));
  
  s2 = createShape(SPHERE,2*r);
  s2.setStroke(color(255,150,0));
  s2.setStrokeWeight(1);
  s2.setFill(color(255,150,0));
  
  a1 = createShape(BOX,w,a,thickness);
  a1.setStroke(color(255,150,0));
  a1.setStrokeWeight(1);
  a1.setFill(color(255,150,0));
  
  a2 = createShape(BOX,w,b,thickness);
  a2.setStroke(color(255,150,0));
  a2.setStrokeWeight(1);
  a2.setFill(color(255,150,0));
  
  float X,Y,Z=0;
  
  pushMatrix();
  shape(s2);
  
  a1.rotate(radians(t_oa),1,0,0);
  Z = (a*0.5+r)*cos(radians(t_oa-90));
  Y = (a*0.5+r)*sin(radians(t_oa-90));
  a1.rotate(radians(t_z),0,0,1);
  X = Y*cos(radians(t_z-90));
  Y = Y*sin(radians(t_z-90));
  translate(X,Y,Z);
  shape(a1);
  
  Z = (0.5*a+r)*cos(radians(t_oa-90));
  Y = (0.5*a+r)*sin(radians(t_oa-90));
  X = Y*cos(radians(t_z-90));
  Y = Y*sin(radians(t_z-90));
  translate(X,Y,Z);
  shape(s1);
  
  t_ab = -180+(t_oa+t_ab);
  a2.rotate(radians(t_ab),1,0,0);
  Z = (b*0.5+r)*cos(radians(t_ab-90));
  Y = (b*0.5+r)*sin(radians(t_ab-90));
  a2.rotate(radians(t_z),0,0,1);
  X = Y*cos(radians(t_z-90));
  Y = Y*sin(radians(t_z-90));
  translate(X,Y,Z);
  
  shape(a2);
  
  popMatrix();
}

void mouseWheel(MouseEvent event) {
  if (event.getCount()==-1){
    scaleLevel += 0.2;
    x += 10;}
  else if (event.getCount()==1){
      x-=10;
  }
}

float rotateX,rotateY,rotateXToUse,rotateYToUse;
void mouseMoved() {
 
  //This method controls rotateXToUse and rotateYToUse variable
 
  rotateX += mouseX-pmouseX; // internal rotate data 
  rotateY += mouseY-pmouseY;
 
  rotateXToUse = (map(rotateX, 0, width, -180, 180)); // rotate data to use
  rotateYToUse = (map(rotateY, 0, height, PI, -PI));
}
