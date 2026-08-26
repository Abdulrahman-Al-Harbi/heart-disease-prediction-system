# استيراد Flask لإنشاء واجهة برمجية API وعرض صفحات HTML
from flask import Flask, request, jsonify, render_template

# استيراد pandas لتحويل بيانات المريض إلى DataFrame
import pandas as pd

# استيراد joblib لتحميل نموذج تعلم الآلة المحفوظ
import joblib


# إنشاء تطبيق Flask
app = Flask(__name__)


# تحميل النموذج المدرب مسبقًا
model = joblib.load("heart_disease_model.pkl")


# تحديد الحقول المطلوبة من بيانات المريض
REQUIRED_FIELDS = [
    "age",
    "sex",
    "dataset",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalch",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal"
]


# تحديد القيم المقبولة للمتغيرات الفئوية
ALLOWED_VALUES = {
    "sex": [
        "Male",
        "Female"
    ],

    "dataset": [
        "Cleveland",
        "Hungary",
        "Switzerland",
        "VA Long Beach"
    ],

    "cp": [
        "typical angina",
        "atypical angina",
        "non-anginal",
        "asymptomatic"
    ],

    "fbs": [
        True,
        False
    ],

    "restecg": [
        "normal",
        "lv hypertrophy",
        "st-t abnormality"
    ],

    "exang": [
        True,
        False
    ],

    "slope": [
        "upsloping",
        "flat",
        "downsloping"
    ],

    "thal": [
        "normal",
        "fixed defect",
        "reversable defect"
    ]
}


# تحديد الحدود البرمجية للمتغيرات الرقمية
# هذه الحدود تستخدم للتحقق من صحة الإدخال وليست حدودًا للتشخيص الطبي
NUMERIC_RANGES = {
    "age": (1, 120),
    "trestbps": (50, 250),
    "chol": (50, 700),
    "thalch": (50, 250),
    "oldpeak": (-10, 20),
    "ca": (0, 3)
}


# إنشاء المسار الرئيسي لعرض واجهة النظام
@app.route("/", methods=["GET"])
def home():

    # عرض صفحة HTML الرئيسية
    return render_template("index.html")


# إنشاء مسار التنبؤ بحالة المريض
@app.route("/predict", methods=["POST"])
def predict():

    # التحقق من أن الطلب يحتوي على بيانات بصيغة JSON
    if not request.is_json:
        return jsonify({
            "error": "يجب إرسال البيانات بصيغة JSON"
        }), 415


    # الحصول على بيانات المريض
    data = request.get_json()


    # التحقق من وجود البيانات
    if not data:
        return jsonify({
            "error": "لم يتم إرسال بيانات المريض"
        }), 400


    # البحث عن الحقول المطلوبة غير الموجودة
    missing_fields = [
        field
        for field in REQUIRED_FIELDS
        if field not in data
    ]


    # التحقق من وجود حقول ناقصة
    if missing_fields:
        return jsonify({
            "error": "توجد حقول مطلوبة مفقودة",
            "missing_fields": missing_fields
        }), 400


    # البحث عن الحقول الإضافية غير المطلوبة
    extra_fields = [
        field
        for field in data
        if field not in REQUIRED_FIELDS
    ]


    # التحقق من وجود حقول غير معروفة
    if extra_fields:
        return jsonify({
            "error": "تم إرسال حقول غير معروفة",
            "extra_fields": extra_fields
        }), 400


    # التحقق من القيم الفئوية
    for field, allowed_values in ALLOWED_VALUES.items():

        # الحصول على قيمة الحقل
        value = data[field]

        # التحقق من أن القيمة ضمن القيم المقبولة
        if value not in allowed_values:
            return jsonify({
                "error": f"قيمة غير صحيحة للحقل {field}",
                "allowed_values": allowed_values,
                "received_value": value
            }), 400


    # التحقق من المتغيرات الرقمية
    for field, (minimum, maximum) in NUMERIC_RANGES.items():

        # الحصول على القيمة
        value = data[field]

        # التحقق من أن القيمة رقمية
        if not isinstance(value, (int, float)) or isinstance(value, bool):

            return jsonify({
                "error": f"يجب أن تكون قيمة {field} رقمية",
                "received_value": value
            }), 400


        # التحقق من أن القيمة تقع ضمن النطاق المحدد
        if value < minimum or value > maximum:

            return jsonify({
                "error": f"قيمة {field} خارج النطاق المسموح",
                "minimum": minimum,
                "maximum": maximum,
                "received_value": value
            }), 400


    # تحويل بيانات المريض إلى DataFrame
    patient_data = pd.DataFrame([data])


    try:

        # إجراء التنبؤ باستخدام النموذج
        prediction = model.predict(patient_data)[0]


        # حساب احتمالية الإصابة بالمرض
        probability = model.predict_proba(patient_data)[0][1]


        # تحديد مستوى الخطورة بناءً على الاحتمالية
        if probability < 0.30:

            risk_level = "منخفض"

        elif probability < 0.70:

            risk_level = "متوسط"

        else:

            risk_level = "مرتفع"


        # تحديد نتيجة التنبؤ
        if prediction == 1:

            diagnosis = "يوجد احتمال للإصابة بمرض القلب"

        else:

            diagnosis = "لا يوجد احتمال مرتفع للإصابة بمرض القلب"


        # إرجاع نتيجة التنبؤ
        return jsonify({

            "prediction": int(prediction),

            "probability": round(
                float(probability),
                4
            ),

            "risk_level": risk_level,

            "diagnosis": diagnosis

        })


    except Exception as error:

        # إرجاع رسالة خطأ في حالة حدوث مشكلة أثناء التنبؤ
        return jsonify({

            "error": "حدث خطأ أثناء معالجة بيانات المريض",

            "details": str(error)

        }), 500


# تشغيل التطبيق عند تنفيذ الملف مباشرة
if __name__ == "__main__":

    # تشغيل خادم Flask في وضع التطوير
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )