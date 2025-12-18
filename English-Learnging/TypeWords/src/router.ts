import * as VueRouter from 'vue-router'
import {RouteRecordRaw} from 'vue-router'
import WordsPage from "@/pages/word/WordsPage.vue";
import Layout from "@/pages/layout.vue";
import ArticlesPage from "@/pages/article/ArticlesPage.vue";
import PracticeArticles from "@/pages/article/PracticeArticles.vue";
import DictDetail from "@/pages/word/DictDetail.vue";
import PracticeWords from "@/pages/word/PracticeWords.vue";
import WordTest from "@/pages/word/WordTest.vue";
import BookDetail from "@/pages/article/BookDetail.vue";
import DictList from "@/pages/word/DictList.vue";
import BookList from "@/pages/article/BookList.vue";
import Setting from "@/pages/setting/Setting.vue";
import Login from "@/pages/user/login.vue";
import User from "@/pages/user/User.vue";
import VipIntro from "@/pages/user/VipIntro.vue";
import Feedback from "@/pages/feedback.vue";
import PasswordLogin from "@/pages/auth/PasswordLogin.vue";
import { useAuthStore } from "@/stores/auth.ts";
// import { useAuthStore } from "@/stores/user.ts";

export const routes: RouteRecordRaw[] = [
  {
    path: '/',
    component: Layout,
    children: [
      {path: '/', redirect: '/words'},
      {path: 'words', component: WordsPage},
      {path: 'word', redirect: '/words'},
      {path: 'practice-words/:id', component: PracticeWords},
      {path: 'word-test/:id', component: WordTest},
      {path: 'study-word', redirect: '/words'},
      {path: 'dict-list', component: DictList},
      {path: 'dict-detail', component: DictDetail},

      {path: 'articles', component: ArticlesPage},
      {path: 'article', redirect: '/articles'},
      {path: 'practice-articles/:id', component: PracticeArticles},
      {path: 'study-article', redirect: '/articles'},
      {path: 'book-detail', component: BookDetail},
      {path: 'book-list', component: BookList},

      {path: 'login', component: Login},
      {path: 'user', component: User},
      {path: 'vip', component: VipIntro},

      {path: 'setting', component: Setting},
      {path: 'feedback', component: Feedback},

    ]
  },
  {path: '/auth/password-login', component: PasswordLogin},
  {path: '/batch-edit-article', component: () => import("@/pages/article/BatchEditArticlePage.vue")},
  {path: '/test', component: () => import("@/pages/test/test.vue")},
  {path: '/:pathMatch(.*)*', redirect: '/words'},
]

const router = VueRouter.createRouter({
  history: VueRouter.createWebHistory(import.meta.env.VITE_ROUTE_BASE),
  // history: VueRouter.createWebHashHistory(),
  routes,
  scrollBehavior(to, from, savedPosition) {
    // console.log('savedPosition', savedPosition)
    if (savedPosition) {
      return savedPosition
    } else {
      return {top: 0}
    }
  },
})

// 路由守卫
router.beforeEach((to: any, from: any) => {
  const authStore = useAuthStore()
  
  // 初始化鉴权状态
  authStore.init()
  
  // 公共路由，不需要密码验证
  const publicRoutes = ['/auth/password-login']
  
  // 如果目标路由是公共路由，直接放行
  if (publicRoutes.includes(to.path)) {
    return true
  }
  
  // 如果用户未认证，跳转到密码登录页
  if (!authStore.checkAuth()) {
    return {
      path: '/auth/password-login',
      query: { redirect: to.fullPath }
    }
  }
  
  return true
  // console.log('beforeEach-to',to.path)
  // console.log('beforeEach-from',from.path)
  // const runtimeStore = useRuntimeStore()
  //
  // //footer下面的5个按钮，对跳不要用动画
  // let noAnimation = [
  //   '/pc/practice',
  //   '/pc/dict',
  //   '/mobile',
  //   '/'
  // ]
  //
  // if (noAnimation.indexOf(from.path) !== -1 && noAnimation.indexOf(to.path) !== -1) {
  //   return true
  // }
  //
  // const toDepth = routes.findIndex(v => v.path === to.path)
  // const fromDepth = routes.findIndex(v => v.path === from.path)
  // // const fromDepth = routeDeep.indexOf(from.path)
  //
  // if (toDepth > fromDepth) {
  //   if (to.matched && to.matched.length) {
  //     let def = to.matched[0].components.default
  //     let toComponentName = def.name ?? def.__name
  //     runtimeStore.updateExcludeRoutes({type: 'remove', value: toComponentName})
  //     // console.log('删除', toComponentName)
  //     // console.log('前进')
  //     // console.log('删除', toComponentName)
  //   }
  // } else {
  //   if (from.matched && from.matched.length) {
  //     let def = from.matched[0].components.default
  //     let fromComponentName = def.name ?? def.__name
  //     runtimeStore.updateExcludeRoutes({type: 'add', value: fromComponentName})
  //     // console.log('添加', fromComponentName)
  //     // console.log('后退')
  //   }
  // }
  // ...
  // 返回 false 以取消导航
  // return true
})


export default router
